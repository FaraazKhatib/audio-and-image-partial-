"""Grad-CAM heatmaps, self contained so it adds no dependency.

Explainability exists here to keep the tool honest rather than to decorate it. A high score
whose heat sits on a watermark, a border or a background texture is a tell that the detector
latched onto a dataset artefact, and that is worth seeing before trusting a verdict.

Every failure path returns None. Hooking internals of arbitrary checkpoints is inherently
fragile, and a missing heatmap must never break an analysis.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

_ANCHORS = np.array(
    [
        [0.0, 0, 0, 0],
        [0.25, 40, 20, 110],
        [0.5, 150, 30, 110],
        [0.75, 240, 100, 40],
        [1.0, 255, 240, 120],
    ]
)


def _colorize(mask: np.ndarray) -> np.ndarray:
    stops = _ANCHORS[:, 0]
    out = np.zeros((*mask.shape, 3), dtype=np.float32)
    for channel in range(3):
        out[..., channel] = np.interp(mask, stops, _ANCHORS[:, channel + 1])
    return out.astype(np.uint8)


def _to_grid(tensor) -> np.ndarray | None:
    """Reduce a captured activation or gradient to a 2D spatial grid.

    Covers the three shapes these backbones emit: channels first conv maps, Swin style
    B H W C blocks, and flat B N C token sequences.
    """
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 4:
        b, d1, d2, d3 = array.shape
        if d1 <= 4 * max(d2, d3) and d1 >= d2 and d1 >= d3:
            return array[0]
        return np.transpose(array[0], (2, 0, 1))
    if array.ndim == 3:
        _, tokens, channels = array.shape
        side = int(round(tokens**0.5))
        if side * side == tokens:
            return np.transpose(array[0].reshape(side, side, channels), (2, 0, 1))
        if side * side == tokens - 1:
            trimmed = array[0][1:]
            return np.transpose(trimmed.reshape(side, side, channels), (2, 0, 1))
    return None


def grad_cam(
    detector,
    image: Image.Image,
    alpha: float = 0.5,
) -> str | None:
    """Return a base64 PNG overlay, or None when the model cannot be hooked."""
    target_name = detector.spec.gradcam_target
    if not detector.ready or not target_name:
        return None

    try:
        import torch
    except ImportError:
        return None

    model = detector._model
    module = dict(model.named_modules()).get(target_name)
    if module is None:
        module = _guess_target(model)
    if module is None:
        return None

    captured: dict[str, object] = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output[0] if isinstance(output, tuple) else output

    def backward_hook(_module, _grad_in, grad_out):
        captured["gradient"] = grad_out[0]

    handles = [
        module.register_forward_hook(forward_hook),
        module.register_full_backward_hook(backward_hook),
    ]

    try:
        inputs = detector._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(detector.device) for k, v in inputs.items()}
        if detector.device == "cuda" and detector._torch_dtype() == torch.float16:
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(**inputs).logits.float()
            target = sum(logits[0, i] for i in detector._fake_indices)
            target.backward()

        activation = captured.get("activation")
        gradient = captured.get("gradient")
        if activation is None or gradient is None:
            return None

        act_grid = _to_grid(activation)
        grad_grid = _to_grid(gradient)
        if act_grid is None or grad_grid is None or act_grid.shape != grad_grid.shape:
            return None

        weights = grad_grid.mean(axis=(1, 2), keepdims=True)
        cam = np.maximum((weights * act_grid).sum(axis=0), 0.0)
        if cam.max() <= 0 or not np.isfinite(cam).all():
            return None
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)

        return _render(image, cam, alpha)
    except Exception:
        return None
    finally:
        for handle in handles:
            handle.remove()
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            pass


def _guess_target(model):
    """Last normalisation or conv layer, used when the configured name does not exist."""
    candidate = None
    for name, module in model.named_modules():
        kind = type(module).__name__
        if kind in ("LayerNorm", "Conv2d", "BatchNorm2d") and "embed" not in name.lower():
            candidate = module
    return candidate


def _render(image: Image.Image, cam: np.ndarray, alpha: float) -> str:
    heat = Image.fromarray(_colorize(cam)).resize(image.size, Image.Resampling.BICUBIC)
    blended = Image.blend(image.convert("RGB"), heat, alpha)
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
