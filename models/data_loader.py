"""Compatibility wrapper for the leakage-safe dataset implementation."""

try:
    from fit_pipeline import MultimodalClinicalDataset as MultiModalClinicalDataset
except ImportError as exc:  # pragma: no cover - import-time safety for standalone use
    raise ImportError(
        "MultiModalClinicalDataset now lives in fit_pipeline.py so that all "
        "training, validation, and test entry points share one split protocol."
    ) from exc
