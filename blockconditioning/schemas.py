from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SalientObjectDescriptions(BaseModel):
    """Structured output returned by the OpenAI vision call."""

    objects: Annotated[list[str], Field(max_length=3)]

    @field_validator("objects")
    @classmethod
    def validate_descriptions(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            description = " ".join(value.strip().split())
            if not description.lower().startswith("a "):
                raise ValueError(f"Object description must start with 'a ': {value!r}")
            if len(description.split()) > 10:
                raise ValueError(f"Object description exceeds 10 words: {value!r}")
            normalized.append(description)
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError("Object descriptions must be unique")
        return normalized


class BoundingBox2D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class BoundingBox3D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    corners: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


class FrameObjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_frame_index: int
    processed_frame_index: int
    original_frame_index: int
    timestamp_seconds: float
    box_3d_world: BoundingBox3D | None
    raw_point_count: int
    filtered_point_count: int


class ObjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    description: str
    first_frame_bbox_xyxy: BoundingBox2D
    isolated_image: str
    frames: list[FrameObjectMetadata]


class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_video: str
    processed_video: str
    first_frame: str
    fps: float
    frame_count: int
    width: int
    height: int
    analysis_processed_frame_indices: list[int]
    objects: list[ObjectMetadata]


@dataclass(slots=True)
class ProcessedVideo:
    source_path: Path
    frames: np.ndarray
    original_frame_indices: np.ndarray
    timestamps_seconds: np.ndarray
    fps: float

    @property
    def height(self) -> int:
        return int(self.frames.shape[1])

    @property
    def width(self) -> int:
        return int(self.frames.shape[2])


@dataclass(slots=True)
class ObjectTrack:
    description: str
    masks: np.ndarray
    scores: np.ndarray
    sam_object_id: int


@dataclass(slots=True)
class DepthResult:
    depth: np.ndarray
    confidence: np.ndarray | None
    intrinsics: np.ndarray
    extrinsics_world_to_camera: np.ndarray
    processed_images: np.ndarray


@dataclass(slots=True)
class ObjectGeometry:
    description: str
    first_frame_bbox_xyxy: tuple[int, int, int, int]
    boxes: list[BoundingBox3D | None]
    raw_point_counts: list[int]
    filtered_point_counts: list[int]


@dataclass(slots=True)
class IsolatedObject:
    description: str
    source_crop_512: np.ndarray
    isolated_image_256: np.ndarray
