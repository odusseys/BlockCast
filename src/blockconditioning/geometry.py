from __future__ import annotations

import cv2
import numpy as np

from blockconditioning.config import GeometryConfig
from blockconditioning.schemas import (
    BoundingBox3D,
    DepthResult,
    ObjectGeometry,
    ObjectTrack,
)


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    if pixels <= 0:
        return binary.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(binary, kernel, iterations=pixels).astype(bool)


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("Cannot compute a 2D box from an empty mask")
    # x_max and y_max are exclusive, matching NumPy/PIL crop semantics.
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def unproject_masked_depth_to_world(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0)
    rows, columns = np.nonzero(valid)
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth[rows, columns].astype(np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera_points = np.column_stack(
        (
            (columns - cx) * z / fx,
            (rows - cy) * z / fy,
            z,
            np.ones_like(z),
        )
    )
    camera_to_world = np.linalg.inv(world_to_camera)
    world_points = (camera_to_world @ camera_points.T).T[:, :3]
    return world_points.astype(np.float32)


def fast_mad_outlier_filter(
    points: np.ndarray,
    *,
    z_threshold: float,
    epsilon: float,
) -> np.ndarray:
    """Linear-time, vectorized robust filter using coordinate-wise MAD scores."""

    if len(points) < 4:
        return points
    median = np.median(points, axis=0)
    mad = np.median(np.abs(points - median), axis=0)
    robust_scale = 1.4826 * mad
    stable_axis = robust_scale > epsilon
    scores = np.zeros_like(points, dtype=np.float32)
    scores[:, stable_axis] = (
        np.abs(points[:, stable_axis] - median[stable_axis])
        / robust_scale[stable_axis]
    )
    return points[np.all(scores <= z_threshold, axis=1)]


def box_from_points(points: np.ndarray) -> BoundingBox3D:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    x0, y0, z0 = minimum.tolist()
    x1, y1, z1 = maximum.tolist()
    corners = (
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    )
    return BoundingBox3D(
        minimum=(x0, y0, z0),
        maximum=(x1, y1, z1),
        corners=corners,
    )


def compute_object_geometry(
    tracks: list[ObjectTrack],
    depth_result: DepthResult,
    config: GeometryConfig,
) -> list[ObjectGeometry]:
    height, width = depth_result.depth.shape[1:]
    geometries: list[ObjectGeometry] = []

    for track in tracks:
        boxes: list[BoundingBox3D | None] = []
        raw_counts: list[int] = []
        filtered_counts: list[int] = []
        first_bbox = mask_bbox_xyxy(track.masks[0])

        for frame_index, mask in enumerate(track.masks):
            eroded = erode_mask(mask, config.mask_erosion_pixels)
            depth_mask = resize_mask(eroded, height, width)
            points = unproject_masked_depth_to_world(
                depth_result.depth[frame_index],
                depth_mask,
                depth_result.intrinsics[frame_index],
                depth_result.extrinsics_world_to_camera[frame_index],
            )
            filtered = fast_mad_outlier_filter(
                points,
                z_threshold=config.mad_z_threshold,
                epsilon=config.mad_epsilon,
            )
            raw_counts.append(len(points))
            filtered_counts.append(len(filtered))
            boxes.append(
                box_from_points(filtered)
                if len(filtered) >= config.minimum_points
                else None
            )

        geometries.append(
            ObjectGeometry(
                description=track.description,
                first_frame_bbox_xyxy=first_bbox,
                boxes=boxes,
                raw_point_counts=raw_counts,
                filtered_point_counts=filtered_counts,
            )
        )
    return geometries

