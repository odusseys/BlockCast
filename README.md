# Block Conditioning

A modular PyTorch preprocessing pipeline for turning videos into object tracks,
per-frame 3D boxes, and isolated object-conditioning images.

The input layout is:

```text
dataset/
└── videos/
    ├── example_01.mp4
    └── example_02.mov
```

The default output layout is:

```text
dataset/
└── blockconditioning_processed/
    └── example_01/
        ├── first_frame.png
        ├── video.mp4
        ├── metadata.json
        └── objects/
            ├── object_00.png
            └── object_01.png
```

Each input is sampled at timestamps `0/16, 1/16, ..., 96/16` seconds. Every
selected frame is resized to 256 pixels high while preserving its aspect ratio.
Thirty indices spanning those 97 frames are used for SAM3 and DA3.

## Pipeline stages

1. Decode 97 frames at 16 fps and resize to 256p.
2. Send the first frame to the OpenAI Responses API at low image detail and
   obtain zero to three structured descriptions beginning with `a `.
3. Run text-prompted `Sam3VideoModel` inference on the 30-frame clip. When a
   prompt produces multiple instances, retain the track with the highest
   first-frame confidence (falling back to best clip confidence).
4. Run DA3 jointly on the same 30 frames to obtain depth, intrinsics, and
   world-to-camera extrinsics.
5. Erode every mask by two pixels, unproject its valid depth pixels to the DA3
   world frame, reject outliers with a vectorized coordinate-wise median
   absolute deviation filter, and compute the exact axis-aligned bounds of the
   remaining points.
6. Crop each first-frame mask box with 16 pixels of padding, scale and
   letterbox it to 512×512, edit it with FLUX.2 Klein using
   `isolate the ...`, center-square the result, and downscale it to 256×256.
7. Save one subfolder and one metadata JSON document per input video.

Heavy model adapters load lazily, move only the active stage to the requested
accelerator, and move it back to CPU afterward. This avoids placing SAM3, DA3,
and FLUX on GPU simultaneously.

The 3D metadata uses DA3's shared world coordinate frame. A box is `null` when
erosion, depth validity, and filtering leave fewer than the configured minimum
number of points. Two-dimensional boxes use `[x_min, y_min, x_max, y_max]`
semantics with exclusive maxima.

## Environment

The scaffold deliberately does not download models. Before running it, install
the package dependencies and install
[Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3) from its
own checkout according to its repository instructions. SAM3 may require an
accepted Hugging Face license and `HF_TOKEN`. OpenAI uses `OPENAI_API_KEY`.

The requested OpenAI configuration is represented literally as:

```python
OpenAIConfig(
    model="luna",
    reasoning_effort="low",
    image_detail="low",
)
```

`luna` is treated as an account-specific model alias because it is not a public
OpenAI API model slug. If the target account exposes it under another name,
replace `PipelineConfig.openai.model` while leaving low reasoning and low image
detail unchanged.

The model adapters follow the primary interfaces documented by
[OpenAI image inputs](https://developers.openai.com/api/docs/guides/images-vision),
[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[SAM3 Video](https://huggingface.co/docs/transformers/model_doc/sam3_video),
[Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3), and
[FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

## Intended invocation

After installing the local project and the model packages:

```bash
blockconditioning /absolute/path/to/dataset
```

Useful options:

```bash
blockconditioning /absolute/path/to/dataset \
  --device cuda \
  --output-folder blockconditioning_processed \
  --video /absolute/path/to/dataset/videos/example_01.mp4
```

The notebook at `notebooks/debug_pipeline.ipynb` selects four random videos and
runs each non-saving stage in a separate cell. It displays:

- each processed 97-frame video;
- its first frame and object descriptions;
- colored SAM3 masks beside the corresponding sampled source frames;
- colored projected 3D boxes on black beside the sampled source frames; and
- each padded object crop beside its FLUX-isolated image.
