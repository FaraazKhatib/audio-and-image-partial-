"""Face detection for the face forgery pathway. OpenCV only, no torch.

Tries YuNet first because it is far more reliable on non frontal and small faces, then falls
back to the Haar cascade bundled with OpenCV so the pathway still works with no downloads at
all. If neither is available the pathway reports itself disabled rather than pretending no
faces were present, which would look identical to a clean result.

The YuNet file is validated before use, not just checked for existence. opencv_zoo tracks it with
Git LFS, so fetching it from raw.githubusercontent.com yields a 131 byte text pointer rather than
the model. That file passes is_file, passes any "already downloaded" check, and then fails deep
inside cv2 with an error that reads like a corrupt model. It shipped that way here and silently
demoted every request to Haar, which is why the face pathway appeared to never find anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

# The real 2023mar model is about 232 KB. Anything under this is not a model, whatever it is.
YUNET_MIN_BYTES = 50_000


def yunet_file_problem(path: Path) -> str | None:
    """Describe why this file cannot be a YuNet model, or None if it looks usable.

    Kept separate from the detector so the download script can validate what it fetched with the
    identical check, rather than the two drifting apart.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(len(LFS_POINTER_PREFIX))
    except OSError as exc:
        return f"cannot be read: {exc}"

    if head.startswith(LFS_POINTER_PREFIX):
        return (
            f"{path} is a Git LFS pointer, not a model. It is {size} bytes of text. Fetch it from "
            f"media.githubusercontent.com rather than raw.githubusercontent.com."
        )
    if size < YUNET_MIN_BYTES:
        return f"{path} is only {size} bytes, far too small to be the YuNet model."
    return None


@dataclass
class Face:
    box: tuple[int, int, int, int]
    score: float

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def as_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "score": round(self.score, 3)}


@dataclass
class FaceDetection:
    faces: list[Face]
    backend: str
    available: bool
    note: str = ""


class FaceDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = "none"
        self.note = ""
        self._yunet = None
        self._cascade = None

    def load(self) -> None:
        mode = self.settings.face_detector
        if mode == "off" or not self.settings.enable_faces:
            self.backend = "disabled"
            self.note = "Face pathway disabled by configuration."
            return

        try:
            import cv2
        except ImportError:
            self.backend = "none"
            self.note = "OpenCV missing, so the face pathway is unavailable."
            return

        if mode in ("auto", "yunet"):
            path = Path(self.settings.yunet_path)
            problem = yunet_file_problem(path) if path.is_file() else f"not found at {path}"

            if problem is None and hasattr(cv2, "FaceDetectorYN"):
                try:
                    self._yunet = cv2.FaceDetectorYN.create(
                        str(path), "", (320, 320), self.settings.face_score_threshold, 0.3, 5000
                    )
                    self.backend = "yunet"
                    return
                except Exception as exc:
                    self.note = f"YuNet failed to initialise: {exc}. "
            elif problem is None:
                self.note = "This OpenCV build has no FaceDetectorYN. "
            elif mode == "yunet":
                self.backend = "none"
                self.note = f"YuNet unusable: {problem}"
                return
            else:
                self.note = (
                    f"YuNet unusable, falling back to Haar: {problem} "
                    f"Run scripts/download_models.py to fetch it properly. Haar misses turned and "
                    f"small faces, so face swaps can go undetected entirely. "
                )

        if mode in ("auto", "haar"):
            try:
                cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(str(cascade_path))
                if cascade.empty():
                    raise RuntimeError(f"cascade empty at {cascade_path}")
                self._cascade = cascade
                self.backend = "haar"
                self.note += "Haar cascade only finds roughly frontal faces."
                return
            except Exception as exc:
                self.note += f"Haar cascade unavailable: {exc}"

        self.backend = "none"

    @property
    def ready(self) -> bool:
        return self.backend in ("yunet", "haar")

    def detect(self, image: Image.Image) -> FaceDetection:
        if not self.ready:
            return FaceDetection([], self.backend, False, self.note)

        import cv2

        rgb = np.asarray(image, dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces: list[Face] = []

        try:
            if self.backend == "yunet":
                height, width = bgr.shape[:2]
                self._yunet.setInputSize((width, height))
                _, detections = self._yunet.detect(bgr)
                for row in detections if detections is not None else []:
                    x, y, w, h = (int(v) for v in row[:4])
                    faces.append(Face((x, y, x + w, y + h), float(row[-1])))
            else:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                boxes = self._cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(self.settings.face_min_size, self.settings.face_min_size),
                )
                for x, y, w, h in boxes:
                    faces.append(Face((int(x), int(y), int(x + w), int(y + h)), 1.0))
        except Exception as exc:
            return FaceDetection([], self.backend, False, f"Face detection failed: {exc}")

        min_size = self.settings.face_min_size
        faces = [
            f
            for f in faces
            if (f.box[2] - f.box[0]) >= min_size and (f.box[3] - f.box[1]) >= min_size
        ]
        faces.sort(key=lambda f: f.area, reverse=True)
        return FaceDetection(faces[: self.settings.max_faces], self.backend, True, self.note)
