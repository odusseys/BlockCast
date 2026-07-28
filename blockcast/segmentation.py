from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from blockcast.config import Sam3Config
from blockcast.schemas import ObjectTrack


def torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _as_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    elif isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        value = [value]
    return [int(item) for item in value]


def _clear_accelerator_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class Sam3VideoSegmenter:
    """Text-prompted SAM3 video adapter that retains one track per prompt."""

    def __init__(
        self,
        config: Sam3Config,
        *,
        device: str,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = torch_dtype(config.dtype)
        if (model is None) != (processor is None):
            raise ValueError("Inject both the SAM3 model and processor, or neither")
        self.model = None if model is None else model.eval()
        self.processor = processor

    def _ensure_loaded(self) -> None:
        if self.model is None:
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            self.model = Sam3VideoModel.from_pretrained(
                self.config.model_id,
                torch_dtype=self.dtype,
            ).eval()
            self.processor = Sam3VideoProcessor.from_pretrained(
                self.config.model_id
            )
        self.model.to(self.device)

    def segment(
        self,
        frames_rgb: np.ndarray,
        descriptions: list[str],
    ) -> list[ObjectTrack]:
        if not descriptions:
            return []
        self._ensure_loaded()
        assert self.model is not None
        assert self.processor is not None
        try:
            session = self.processor.init_video_session(
                video=frames_rgb,
                inference_device=self.device,
                inference_state_device=self.config.inference_state_device,
                processing_device=self.config.processing_device,
                video_storage_device=self.config.video_storage_device,
                dtype=self.dtype,
            )
            self.processor.add_text_prompt(session, descriptions)

            outputs: dict[int, Mapping[str, Any]] = {}
            with torch.inference_mode():
                iterator = self.model.propagate_in_video_iterator(
                    inference_session=session,
                    max_frame_num_to_track=len(frames_rgb) - 1,
                )
                for model_output in iterator:
                    processed = self.processor.postprocess_outputs(
                        session,
                        model_output,
                    )
                    outputs[int(model_output.frame_idx)] = {
                        "object_ids": _as_ids(processed["object_ids"]),
                        "scores": _as_numpy(processed["scores"]),
                        "masks": _as_numpy(processed["masks"]),
                        "prompt_to_obj_ids": {
                            str(prompt): _as_ids(ids)
                            for prompt, ids in processed[
                                "prompt_to_obj_ids"
                            ].items()
                        },
                    }

            return self._select_top_tracks(
                outputs=outputs,
                descriptions=descriptions,
                frame_shape=frames_rgb.shape[1:3],
                frame_count=len(frames_rgb),
            )
        finally:
            self.model.to("cpu")
            _clear_accelerator_cache(self.device)

    def _select_top_tracks(
        self,
        *,
        outputs: dict[int, Mapping[str, Any]],
        descriptions: list[str],
        frame_shape: tuple[int, int],
        frame_count: int,
    ) -> list[ObjectTrack]:
        prompt_candidates: dict[str, set[int]] = defaultdict(set)
        score_history: dict[int, dict[int, float]] = defaultdict(dict)
        mask_history: dict[int, dict[int, np.ndarray]] = defaultdict(dict)

        for frame_index, output in outputs.items():
            ids = _as_ids(output.get("object_ids", []))
            scores = _as_numpy(output.get("scores", np.zeros(len(ids)))).reshape(-1)
            masks = _as_numpy(
                output.get(
                    "masks",
                    np.zeros((len(ids), *frame_shape), dtype=np.float32),
                )
            )
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]

            for position, object_id in enumerate(ids):
                score_history[object_id][frame_index] = float(scores[position])
                mask_history[object_id][frame_index] = (
                    masks[position] > self.config.mask_threshold
                )

            prompt_mapping = output.get("prompt_to_obj_ids", {})
            for prompt, object_ids in prompt_mapping.items():
                prompt_candidates[str(prompt)].update(_as_ids(object_ids))

        tracks: list[ObjectTrack] = []
        for description in descriptions:
            candidates = prompt_candidates.get(description, set())
            first_frame_candidates = [
                object_id
                for object_id in candidates
                if mask_history[object_id].get(
                    0,
                    np.zeros(frame_shape, dtype=bool),
                ).any()
            ]
            if not first_frame_candidates:
                continue
            selected_id = max(
                first_frame_candidates,
                key=lambda object_id: self._track_rank(
                    score_history.get(object_id, {})
                ),
            )

            selected_masks: list[np.ndarray] = []
            selected_scores: list[float] = []
            for frame_index in range(frame_count):
                selected_masks.append(
                    mask_history[selected_id].get(
                        frame_index,
                        np.zeros(frame_shape, dtype=bool),
                    )
                )
                selected_scores.append(
                    score_history[selected_id].get(frame_index, float("-inf"))
                )

            tracks.append(
                ObjectTrack(
                    description=description,
                    masks=np.stack(selected_masks),
                    scores=np.asarray(selected_scores, dtype=np.float32),
                    sam_object_id=selected_id,
                )
            )
        return tracks

    @staticmethod
    def _track_rank(frame_scores: dict[int, float]) -> tuple[float, float]:
        """Prefer first-frame confidence, then best confidence in the clip."""

        first_score = frame_scores.get(0, float("-inf"))
        best_score = max(frame_scores.values(), default=float("-inf"))
        return first_score, best_score
