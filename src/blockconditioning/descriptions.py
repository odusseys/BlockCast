from __future__ import annotations

import base64
import io

import numpy as np
from openai import OpenAI
from PIL import Image

from blockconditioning.config import OpenAIConfig
from blockconditioning.schemas import SalientObjectDescriptions


def _png_data_url(frame_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(frame_rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class SalientObjectDescriber:
    def __init__(
        self,
        config: OpenAIConfig,
        *,
        client: OpenAI | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAI()

    def describe(self, first_frame_rgb: np.ndarray) -> list[str]:
        prompt = (
            f"Identify up to the {self.config.max_objects} most visually salient "
            "physical objects in this frame, including people. Return fewer when "
            "fewer distinct salient objects exist. Each description must start "
            "exactly with 'a ', be short and specific, contain at most 10 words, "
            "and distinguish similar instances using visible attributes. Do not "
            "include background regions, materials, body parts, or scene labels."
        )
        response = self.client.responses.parse(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": _png_data_url(first_frame_rgb),
                            "detail": self.config.image_detail,
                        },
                    ],
                }
            ],
            text_format=SalientObjectDescriptions,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI returned no parsed object descriptions")
        return response.output_parsed.objects[: self.config.max_objects]

