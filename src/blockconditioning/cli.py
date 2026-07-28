from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from blockconditioning.config import OutputConfig, PipelineConfig
from blockconditioning.pipeline import VideoPipeline, build_model_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the segmented 3D object-conditioning dataset."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset root containing videos/",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-folder",
        default="blockconditioning_processed",
        help="New folder created below the dataset root.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Process one video instead of every file in dataset/videos.",
    )
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = PipelineConfig(
        dataset_dir=arguments.dataset_dir.expanduser().resolve(),
        device=arguments.device,
    )
    config = replace(
        config,
        output=OutputConfig(folder_name=arguments.output_folder),
    )
    pipeline = VideoPipeline(config, build_model_bundle(config))
    if arguments.video:
        pipeline.process_and_save(arguments.video.expanduser().resolve())
    else:
        pipeline.process_dataset(limit=arguments.limit)


if __name__ == "__main__":
    main()
