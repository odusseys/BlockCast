from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from blockcast.config import PipelineConfig
from blockcast.depth import DA3PointCloudEstimator
from blockcast.descriptions import SalientObjectDescriber
from blockcast.geometry import compute_object_geometry
from blockcast.isolation import KleinObjectIsolator
from blockcast.schemas import (
    BoundingBox2D,
    DepthResult,
    FrameObjectMetadata,
    IsolatedObject,
    ObjectGeometry,
    ObjectMetadata,
    ObjectTrack,
    ProcessedVideo,
    VideoMetadata,
)
from blockcast.segmentation import Sam3VideoSegmenter
from blockcast.video import (
    choose_analysis_indices,
    decode_and_preprocess_video,
    list_videos,
    write_video,
)


@dataclass(slots=True)
class ModelBundle:
    describer: SalientObjectDescriber
    segmenter: Sam3VideoSegmenter
    depth_estimator: DA3PointCloudEstimator
    isolator: KleinObjectIsolator


@dataclass(slots=True)
class VideoStageResults:
    video: ProcessedVideo
    analysis_indices: np.ndarray
    descriptions: list[str]
    tracks: list[ObjectTrack]
    depth: DepthResult
    geometries: list[ObjectGeometry]
    isolated_objects: list[IsolatedObject]


def build_model_bundle(config: PipelineConfig) -> ModelBundle:
    return ModelBundle(
        describer=SalientObjectDescriber(config.openai),
        segmenter=Sam3VideoSegmenter(
            config.sam3,
            device=config.device,
        ),
        depth_estimator=DA3PointCloudEstimator(
            config.da3,
            device=config.device,
        ),
        isolator=KleinObjectIsolator(
            config.klein,
            device=config.device,
        ),
    )


def _safe_stem(path: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return slug or "video"


class VideoPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        models: ModelBundle,
    ) -> None:
        self.config = config
        self.models = models

    def process_stages(self, source_path: Path) -> VideoStageResults:
        video = decode_and_preprocess_video(source_path, self.config.video)
        descriptions = self.models.describer.describe(video.frames[0])
        analysis_indices = choose_analysis_indices(
            self.config.video.frame_count,
            self.config.video.analysis_frame_count,
        )
        analysis_frames = video.frames[analysis_indices]
        tracks = self.models.segmenter.segment(analysis_frames, descriptions)
        depth = self.models.depth_estimator.predict(analysis_frames)
        geometries = compute_object_geometry(
            tracks,
            depth,
            self.config.geometry,
        )
        isolated_objects = self.models.isolator.isolate(
            video.frames[0],
            geometries,
        )
        return VideoStageResults(
            video=video,
            analysis_indices=analysis_indices,
            descriptions=descriptions,
            tracks=tracks,
            depth=depth,
            geometries=geometries,
            isolated_objects=isolated_objects,
        )

    def process_and_save(self, source_path: Path) -> Path:
        results = self.process_stages(source_path)
        destination = self.config.output_dir / _safe_stem(source_path)
        self.save_results(results, destination)
        return destination

    def process_dataset(self, limit: int | None = None) -> list[Path]:
        videos = list_videos(self.config.videos_dir)
        if limit is not None:
            videos = videos[:limit]
        outputs: list[Path] = []
        for video_path in tqdm(videos, desc="videos"):
            outputs.append(self.process_and_save(video_path))
        return outputs

    def save_results(
        self,
        results: VideoStageResults,
        destination: Path,
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        objects_dir = destination / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)

        first_frame_path = destination / "first_frame.png"
        processed_video_path = destination / "video.mp4"
        metadata_path = destination / "metadata.json"
        Image.fromarray(results.video.frames[0]).save(first_frame_path)
        write_video(
            processed_video_path,
            results.video.frames,
            results.video.fps,
            codec=self.config.output.video_codec,
            crf=self.config.output.video_crf,
        )

        object_metadata: list[ObjectMetadata] = []
        isolated_by_description = {
            item.description: item for item in results.isolated_objects
        }
        for object_index, geometry in enumerate(results.geometries):
            object_id = f"object_{object_index:02d}"
            isolated_filename = f"{object_id}.png"
            isolated = isolated_by_description[geometry.description]
            Image.fromarray(isolated.isolated_image_256).save(
                objects_dir / isolated_filename
            )
            x0, y0, x1, y1 = geometry.first_frame_bbox_xyxy

            frames = [
                FrameObjectMetadata(
                    analysis_frame_index=analysis_frame_index,
                    processed_frame_index=int(processed_frame_index),
                    original_frame_index=int(
                        results.video.original_frame_indices[
                            processed_frame_index
                        ]
                    ),
                    timestamp_seconds=float(
                        results.video.timestamps_seconds[processed_frame_index]
                    ),
                    box_3d_world=geometry.boxes[analysis_frame_index],
                    raw_point_count=geometry.raw_point_counts[
                        analysis_frame_index
                    ],
                    filtered_point_count=geometry.filtered_point_counts[
                        analysis_frame_index
                    ],
                )
                for analysis_frame_index, processed_frame_index in enumerate(
                    results.analysis_indices
                )
            ]
            object_metadata.append(
                ObjectMetadata(
                    object_id=object_id,
                    description=geometry.description,
                    first_frame_bbox_xyxy=BoundingBox2D(
                        x_min=x0,
                        y_min=y0,
                        x_max=x1,
                        y_max=y1,
                    ),
                    isolated_image=f"objects/{isolated_filename}",
                    frames=frames,
                )
            )

        metadata = VideoMetadata(
            source_video=str(results.video.source_path),
            processed_video=processed_video_path.name,
            first_frame=first_frame_path.name,
            fps=results.video.fps,
            frame_count=len(results.video.frames),
            width=results.video.width,
            height=results.video.height,
            analysis_processed_frame_indices=[
                int(index) for index in results.analysis_indices
            ],
            objects=object_metadata,
        )
        metadata_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
