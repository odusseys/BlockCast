from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from blockconditioning.config import KleinConfig
from blockconditioning.schemas import IsolatedObject, ObjectGeometry
from blockconditioning.segmentation import torch_dtype


def crop_scale_and_pad(
    frame_rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    padding: int,
    size: int,
    pad_color: tuple[int, int, int],
) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(width, x1 + padding), min(height, y1 + padding)
    crop = Image.fromarray(frame_rgb[y0:y1, x0:x1])

    scale = size / max(crop.size)
    resized_size = (
        max(1, int(round(crop.width * scale))),
        max(1, int(round(crop.height * scale))),
    )
    crop = crop.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=pad_color)
    offset = ((size - crop.width) // 2, (size - crop.height) // 2)
    canvas.paste(crop, offset)
    return np.asarray(canvas)


def klein_prompt(description: str) -> str:
    subject = description.strip()
    if subject.casefold().startswith("a "):
        subject = f"the {subject[2:]}"
    return f"isolate {subject}"


def square_and_resize(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert("RGB")
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    square = image.crop((left, top, left + side, top + side))
    return np.asarray(square.resize((size, size), Image.Resampling.LANCZOS))


class KleinObjectIsolator:
    def __init__(
        self,
        config: KleinConfig,
        *,
        device: str,
        pipeline: Any | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.pipeline = pipeline

    def _ensure_loaded(self) -> None:
        if self.pipeline is None:
            from diffusers import Flux2KleinPipeline

            self.pipeline = Flux2KleinPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch_dtype(self.config.dtype),
            )
        self.pipeline.to(self.device)

    def isolate(
        self,
        first_frame_rgb: np.ndarray,
        geometries: list[ObjectGeometry],
    ) -> list[IsolatedObject]:
        if not geometries:
            return []
        self._ensure_loaded()
        assert self.pipeline is not None
        results: list[IsolatedObject] = []
        try:
            for object_index, geometry in enumerate(geometries):
                source_crop = crop_scale_and_pad(
                    first_frame_rgb,
                    geometry.first_frame_bbox_xyxy,
                    padding=self.config.crop_padding_pixels,
                    size=self.config.input_size,
                    pad_color=self.config.pad_color,
                )
                generator = torch.Generator(device=self.device).manual_seed(
                    self.config.seed + object_index
                )
                output = self.pipeline(
                    prompt=klein_prompt(geometry.description),
                    image=[Image.fromarray(source_crop)],
                    height=self.config.input_size,
                    width=self.config.input_size,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    generator=generator,
                ).images[0]
                results.append(
                    IsolatedObject(
                        description=geometry.description,
                        source_crop_512=source_crop,
                        isolated_image_256=square_and_resize(
                            output,
                            self.config.output_size,
                        ),
                    )
                )
            return results
        finally:
            self.pipeline.to("cpu")
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif (
                self.device.startswith("mps")
                and torch.backends.mps.is_available()
            ):
                torch.mps.empty_cache()
