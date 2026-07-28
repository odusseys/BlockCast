from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoConfig:
    fps: float = 16.0
    frame_count: int = 97
    target_height: int = 256
    analysis_frame_count: int = 30


@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    # "luna" is kept as requested. Override if the account uses another alias.
    model: str = "luna"
    reasoning_effort: str = "low"
    image_detail: str = "low"
    max_objects: int = 3


@dataclass(frozen=True, slots=True)
class Sam3Config:
    model_id: str = "facebook/sam3"
    dtype: str = "bfloat16"
    inference_state_device: str = "cpu"
    processing_device: str = "cpu"
    video_storage_device: str = "cpu"
    mask_threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class DA3Config:
    model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    process_resolution: int = 504
    process_resize_method: str = "upper_bound_resize"
    reference_view_strategy: str = "middle"


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    mask_erosion_pixels: int = 2
    minimum_points: int = 32
    mad_z_threshold: float = 4.5
    mad_epsilon: float = 1e-6


@dataclass(frozen=True, slots=True)
class KleinConfig:
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    dtype: str = "bfloat16"
    crop_padding_pixels: int = 16
    input_size: int = 512
    output_size: int = 256
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    seed: int = 0
    pad_color: tuple[int, int, int] = (255, 255, 255)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    folder_name: str = "blockconditioning_processed"
    video_codec: str = "libx264"
    video_crf: int = 18


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    dataset_dir: Path
    device: str = "cuda"
    video: VideoConfig = field(default_factory=VideoConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    sam3: Sam3Config = field(default_factory=Sam3Config)
    da3: DA3Config = field(default_factory=DA3Config)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    klein: KleinConfig = field(default_factory=KleinConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def videos_dir(self) -> Path:
        return self.dataset_dir / "videos"

    @property
    def output_dir(self) -> Path:
        return self.dataset_dir / self.output.folder_name
