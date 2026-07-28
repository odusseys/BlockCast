from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from blockcast.config import VideoConfig
from blockcast.schemas import ProcessedVideo

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def list_videos(videos_dir: Path) -> list[Path]:
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {videos_dir}")
    return sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def choose_analysis_indices(frame_count: int, analysis_count: int) -> np.ndarray:
    if analysis_count > frame_count:
        raise ValueError("analysis_count cannot exceed frame_count")
    return np.linspace(0, frame_count - 1, analysis_count, dtype=np.int64)


def _resize_to_height(frame_rgb: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    if height == target_height and width % 2 == 0:
        return frame_rgb
    scaled_width = (
        width
        if height == target_height
        else max(2, int(round(width * target_height / height)))
    )
    # H.264 requires even dimensions in the common yuv420p format.
    scaled_width += scaled_width % 2
    interpolation = cv2.INTER_AREA if target_height < height else cv2.INTER_CUBIC
    return cv2.resize(
        frame_rgb,
        (scaled_width, target_height),
        interpolation=interpolation,
    )


def decode_and_preprocess_video(
    video_path: Path,
    config: VideoConfig,
) -> ProcessedVideo:
    """Sample the first 97 target timestamps at 16 fps and resize to 256p."""

    target_times = np.arange(config.frame_count, dtype=np.float64) / config.fps
    frames: list[np.ndarray] = []
    original_indices: list[int] = []
    timestamps: list[float] = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        source_start: float | None = None
        target_index = 0

        for source_index, frame in enumerate(container.decode(stream)):
            absolute_time = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None and frame.time_base is not None
                else source_index / float(stream.average_rate or config.fps)
            )
            if source_start is None:
                source_start = absolute_time
            relative_time = absolute_time - source_start

            while (
                target_index < config.frame_count
                and relative_time + 1e-9 >= target_times[target_index]
            ):
                rgb = frame.to_ndarray(format="rgb24")
                frames.append(_resize_to_height(rgb, config.target_height))
                original_indices.append(source_index)
                timestamps.append(absolute_time)
                target_index += 1

            if target_index == config.frame_count:
                break

    if len(frames) != config.frame_count:
        required_seconds = (config.frame_count - 1) / config.fps
        raise ValueError(
            f"{video_path} produced {len(frames)}/{config.frame_count} frames; "
            f"at least {required_seconds:.3f}s of decodable video is required"
        )

    return ProcessedVideo(
        source_path=video_path,
        frames=np.stack(frames),
        original_frame_indices=np.asarray(original_indices, dtype=np.int64),
        timestamps_seconds=np.asarray(timestamps, dtype=np.float64),
        fps=config.fps,
    )


def write_video(
    path: Path,
    frames: np.ndarray,
    fps: float,
    *,
    codec: str = "libx264",
    crf: int = 18,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=Fraction(str(fps)))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}
        for array in frames:
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
