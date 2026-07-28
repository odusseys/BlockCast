from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from blockconditioning.schemas import (
    BoundingBox3D,
    DepthResult,
    IsolatedObject,
    ObjectGeometry,
    ObjectTrack,
)

OBJECT_COLORS_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 64, 64),
    (64, 220, 96),
    (64, 144, 255),
)

BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _draw_label(
    image_rgb: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color_rgb: tuple[int, int, int],
) -> None:
    cv2.putText(
        image_rgb,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color_rgb,
        1,
        cv2.LINE_AA,
    )


def segmentation_overlay_frames(
    frames_rgb: np.ndarray,
    tracks: list[ObjectTrack],
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    overlays: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames_rgb):
        overlay = frame.copy()
        label_row = 18
        for object_index, track in enumerate(tracks):
            color = np.asarray(
                OBJECT_COLORS_RGB[object_index % len(OBJECT_COLORS_RGB)]
            )
            mask = track.masks[frame_index]
            if mask.any():
                pixels = overlay[mask].astype(np.float32)
                overlay[mask] = (
                    pixels * (1.0 - alpha) + color * alpha
                ).astype(np.uint8)
            _draw_label(
                overlay,
                track.description,
                (8, label_row),
                tuple(int(channel) for channel in color),
            )
            label_row += 18
        overlays.append(overlay)
    return np.stack(overlays)


def side_by_side(
    left_frames_rgb: np.ndarray,
    right_frames_rgb: np.ndarray,
) -> np.ndarray:
    if left_frames_rgb.shape[:3] != right_frames_rgb.shape[:3]:
        raise ValueError(
            f"Video shapes must match: {left_frames_rgb.shape} vs "
            f"{right_frames_rgb.shape}"
        )
    return np.concatenate((left_frames_rgb, right_frames_rgb), axis=2)


def _project_box(
    box: BoundingBox3D,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(box.corners, dtype=np.float64)
    homogeneous = np.column_stack((corners, np.ones(len(corners))))
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    positive_depth = camera[:, 2] > 1e-6
    image_homogeneous = (intrinsics @ camera.T).T
    image_points = image_homogeneous[:, :2] / np.maximum(
        image_homogeneous[:, 2:3],
        1e-6,
    )
    return image_points, positive_depth


def render_3d_boxes_on_black(
    geometries: list[ObjectGeometry],
    depth_result: DepthResult,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    depth_height, depth_width = depth_result.depth.shape[1:]
    rendered: list[np.ndarray] = []

    for frame_index in range(len(depth_result.depth)):
        canvas = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)
        label_row = 18
        for object_index, geometry in enumerate(geometries):
            color = OBJECT_COLORS_RGB[object_index % len(OBJECT_COLORS_RGB)]
            box = geometry.boxes[frame_index]
            if box is not None:
                points, visible = _project_box(
                    box,
                    depth_result.intrinsics[frame_index],
                    depth_result.extrinsics_world_to_camera[frame_index],
                )
                rounded = np.rint(points).astype(np.int32)
                for first, second in BOX_EDGES:
                    if visible[first] and visible[second]:
                        cv2.line(
                            canvas,
                            tuple(rounded[first]),
                            tuple(rounded[second]),
                            color,
                            2,
                            cv2.LINE_AA,
                        )
            _draw_label(canvas, geometry.description, (8, label_row), color)
            label_row += 18

        if (depth_height, depth_width) != (output_height, output_width):
            canvas = cv2.resize(
                canvas,
                (output_width, output_height),
                interpolation=cv2.INTER_LINEAR,
            )
        rendered.append(canvas)
    return np.stack(rendered)


def isolation_comparison_images(
    objects: Sequence[IsolatedObject],
) -> list[tuple[str, np.ndarray]]:
    comparisons: list[tuple[str, np.ndarray]] = []
    for item in objects:
        source = cv2.resize(
            item.source_crop_512,
            (
                item.isolated_image_256.shape[1],
                item.isolated_image_256.shape[0],
            ),
            interpolation=cv2.INTER_AREA,
        )
        comparisons.append(
            (
                item.description,
                np.concatenate((source, item.isolated_image_256), axis=1),
            )
        )
    return comparisons


def display_animation(
    frames_rgb: np.ndarray,
    fps: float,
    *,
    title: str | None = None,
):
    """Return a JS-backed notebook animation without writing a debug video."""

    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.axis("off")
    if title:
        axis.set_title(title)
    artist = axis.imshow(frames_rgb[0])

    def update(frame_index: int):
        artist.set_data(frames_rgb[frame_index])
        return (artist,)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames_rgb),
        interval=1000.0 / fps,
        blit=True,
    )
    html = HTML(animation.to_jshtml())
    plt.close(figure)
    return html


def display_image_pairs(
    comparisons: Sequence[tuple[str, np.ndarray]],
):
    import matplotlib.pyplot as plt

    if not comparisons:
        return None
    figure, axes = plt.subplots(
        len(comparisons),
        1,
        figsize=(8, 3 * len(comparisons)),
        squeeze=False,
    )
    for axis, (description, image) in zip(axes[:, 0], comparisons, strict=True):
        axis.imshow(image)
        axis.set_title(f"{description}: crop (left) / isolated (right)")
        axis.axis("off")
    figure.tight_layout()
    return figure

