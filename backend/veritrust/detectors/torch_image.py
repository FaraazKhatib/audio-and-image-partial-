"""PyTorch image classification wrapper for raw FaceForensics++ models.

This class explicitly loads raw `.pth` checkpoints (specifically EfficientNet-B0 trained on FF++)
that cannot be loaded via the standard Hugging Face pipeline because they lack a config.json.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from ..config import ModelSpec
from .base import Detector, DetectorResult, LoadError


# This Hub identifier is retained in ModelSpec for provenance, but it is served from the raw
# FaceForensics++ checkpoint bundled under backend/scratch rather than from a Transformers layout.
RAW_TORCH_CHECKPOINTS = {
    "Xicor9/efficientnet-b0-ffpp-c23": "efficientnet_b0_ffpp_c23.pth",
}
RAW_TORCH_REPOS = frozenset(RAW_TORCH_CHECKPOINTS)


def uses_raw_torch_checkpoint(spec: ModelSpec) -> bool:
    """Whether ``spec`` requires TorchImageClassifier instead of the HF wrapper."""
    return spec.repo in RAW_TORCH_REPOS


def raw_checkpoint_path(spec: ModelSpec) -> Path:
    """Absolute location for a registered raw checkpoint."""
    filename = RAW_TORCH_CHECKPOINTS.get(spec.repo)
    if filename is None:
        raise ValueError(f"{spec.repo} is not a registered raw PyTorch checkpoint")
    return Path(__file__).resolve().parents[2] / "scratch" / filename


def raw_checkpoint_url(spec: ModelSpec) -> str:
    """Published weight URL for a registered raw checkpoint."""
    return f"https://huggingface.co/{spec.repo}/resolve/main/{RAW_TORCH_CHECKPOINTS[spec.repo]}"


class TorchImageClassifier(Detector):
    def __init__(self, spec: ModelSpec, device: str = "cpu", dtype: str = "float32"):
        self.spec = spec
        self.key = spec.key
        self.kind = spec.kind
        self.device = device
        self.dtype = dtype
        self._model = None
        self._transform = None
        self.repo_used = spec.repo
        self.error = None
        self._fake_index = 0
        self.label_mapping = {"0": "fake", "1": "real"}

    def load(self) -> None:
        try:
            # Resolve relative to backend/, not the process working directory.  The web server
            # normally starts from backend/ but verification, tests, and process managers need not.
            weights_path = raw_checkpoint_path(self.spec)
            if not weights_path.exists():
                raise LoadError(f"{weights_path} not found. Please download the weights first.")

            # Initialize the base EfficientNet-B0 architecture
            model = models.efficientnet_b0(weights=None)
            
            # The FaceForensics++ checkpoint replaces the default 1000-class ImageNet
            # classifier head with a 2-class head, ordered real then fake.
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 2)
            
            # Load the state dict safely
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            self._model = model
            
            # Match this checkpoint's published inference recipe exactly: Resize then ToTensor.
            # Its model card does not apply ImageNet mean/std normalisation; adding it changes the
            # feature distribution and can turn a confident face-forgery read into a confident
            # but meaningless "real" result.
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            self.error = str(e)
            raise LoadError(f"Failed to load PyTorch model: {e}")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, image: Image.Image) -> DetectorResult:
        if not self.ready:
            return DetectorResult(
                key=self.key,
                kind=self.kind,
                p_fake=None,
                error=self.error or "not loaded",
            )
            
        started = time.perf_counter()
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            input_tensor = self._transform(image).unsqueeze(0).to(self.device)
            
            with torch.inference_mode():
                logits = self._model(input_tensor).float()
                
            probs = torch.softmax(logits, dim=-1)[0]
            
            # The exact Xicor9 checkpoint in this registry is 0=Real, 1=Fake.  Do not rely on a
            # generic FF++ convention here: the repository's model card is the source of truth.
            p_fake = float(probs[self._fake_index].item())
            
        except Exception as exc:
            return DetectorResult(
                key=self.key,
                kind=self.kind,
                p_fake=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        return DetectorResult(
            key=self.key,
            kind=self.kind,
            p_fake=min(max(p_fake, 0.0), 1.0),
            latency_ms=(time.perf_counter() - started) * 1000,
            detail=self.spec.notes,
        )

    def describe(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "configured_repo": self.spec.repo,
            "loaded_repo": self.repo_used,
            "ready": self.ready,
            "label_mapping": self.label_mapping,
            "weight": self.spec.weight,
            "notes": self.spec.notes,
            "error": self.error,
        }
