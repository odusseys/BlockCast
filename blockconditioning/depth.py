from __future__ import annotations

from typing import Any

import numpy as np
import torch

from blockconditioning.config import DA3Config
from blockconditioning.schemas import DepthResult


def _homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")
    result = np.broadcast_to(
        np.eye(4, dtype=np.float64),
        (*extrinsics.shape[:-2], 4, 4),
    ).copy()
    result[..., :3, :] = extrinsics
    return result


class DA3PointCloudEstimator:
    def __init__(
        self,
        config: DA3Config,
        *,
        device: str,
        model: Any | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.model = None if model is None else model.eval()

    def _ensure_loaded(self) -> None:
        if self.model is None:
            from depth_anything_3.api import DepthAnything3

            self.model = DepthAnything3.from_pretrained(
                self.config.model_id
            ).eval()
        self.model.to(self.device)

    def predict(self, frames_rgb: np.ndarray) -> DepthResult:
        self._ensure_loaded()
        assert self.model is not None
        try:
            prediction = self.model.inference(
                image=[frame for frame in frames_rgb],
                process_res=self.config.process_resolution,
                process_res_method=self.config.process_resize_method,
                ref_view_strategy=self.config.reference_view_strategy,
            )
            confidence = getattr(prediction, "conf", None)
            return DepthResult(
                depth=np.asarray(prediction.depth, dtype=np.float32),
                confidence=(
                    None
                    if confidence is None
                    else np.asarray(confidence, dtype=np.float32)
                ),
                intrinsics=np.asarray(prediction.intrinsics, dtype=np.float64),
                extrinsics_world_to_camera=_homogeneous_extrinsics(
                    prediction.extrinsics
                ),
                processed_images=np.asarray(prediction.processed_images),
            )
        finally:
            self.model.to("cpu")
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif (
                self.device.startswith("mps")
                and torch.backends.mps.is_available()
            ):
                torch.mps.empty_cache()
