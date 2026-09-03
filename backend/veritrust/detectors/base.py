"""Detector interface shared by every signal source."""

from __future__ import annotations

from dataclasses import dataclass, field


class LoadError(RuntimeError):
    """A checkpoint could not be loaded or its labels could not be resolved safely."""


@dataclass
class DetectorResult:
    key: str
    kind: str
    p_fake: float | None
    latency_ms: float = 0.0
    detail: str = ""
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.error is None and self.p_fake is not None


class Detector:
    """Base class. Loading is lazy and failure is non fatal by design.

    A dead or renamed checkpoint degrades the ensemble to its remaining members instead of
    taking the service down, and /api/v1/models reports exactly what is missing.
    """

    key: str = "detector"
    kind: str = "unknown"

    def load(self) -> None:
        raise NotImplementedError

    @property
    def ready(self) -> bool:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError
