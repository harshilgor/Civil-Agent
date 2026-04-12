"""
OpenCV-backed floor-plan parsing primitives.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency path
    cv2 = None


def cv_available() -> bool:
    return cv2 is not None


def _group_positions(values: list[dict[str, float]], tolerance_px: float = 6.0) -> list[dict[str, float]]:
    if not values:
        return []
    ordered = sorted(values, key=lambda item: item["position_px"])
    grouped = [ordered[0].copy()]
    for item in ordered[1:]:
        current = grouped[-1]
        if abs(float(item["position_px"]) - float(current["position_px"])) <= tolerance_px:
            count = float(current.get("_count", 1.0))
            current["position_px"] = round((float(current["position_px"]) * count + float(item["position_px"])) / (count + 1.0), 1)
            current["thickness_px"] = round(max(float(current["thickness_px"]), float(item["thickness_px"])), 1)
            current["start_px"] = min(float(current["start_px"]), float(item["start_px"]))
            current["end_px"] = max(float(current["end_px"]), float(item["end_px"]))
            current["_count"] = count + 1.0
        else:
            grouped.append(item.copy())
    for item in grouped:
        item.pop("_count", None)
    return grouped


def _segments_to_lines(segments: np.ndarray | None, *, min_length_px: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vertical: list[dict[str, Any]] = []
    horizontal: list[dict[str, Any]] = []
    if segments is None:
        return vertical, horizontal

    for segment in segments:
        x1, y1, x2, y2 = [int(value) for value in segment[0]]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = float((dx**2 + dy**2) ** 0.5)
        if length < min_length_px:
            continue
        if dx <= max(5, int(0.12 * max(dy, 1))):
            vertical.append(
                {
                    "orientation": "vertical",
                    "position_px": round((x1 + x2) / 2.0, 1),
                    "thickness_px": float(max(3, dx + 1)),
                    "start_px": float(min(y1, y2)),
                    "end_px": float(max(y1, y2)),
                }
            )
        elif dy <= max(5, int(0.12 * max(dx, 1))):
            horizontal.append(
                {
                    "orientation": "horizontal",
                    "position_px": round((y1 + y2) / 2.0, 1),
                    "thickness_px": float(max(3, dy + 1)),
                    "start_px": float(min(x1, x2)),
                    "end_px": float(max(x1, x2)),
                }
            )

    return _group_positions(vertical), _group_positions(horizontal)


def _outline_from_contours(binary_inv: np.ndarray, image_width: int, image_height: int) -> tuple[dict[str, int], list[tuple[int, int]], int]:
    coords = cv2.findNonZero(binary_inv) if cv2 is not None else None
    if coords is None:
        bbox = {"x_min": 0, "y_min": 0, "x_max": image_width, "y_max": image_height}
        outline = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]
        return bbox, outline, 0

    x, y, w, h = cv2.boundingRect(coords)
    contours, hierarchy = cv2.findContours(binary_inv.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        outline = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        if len(outline) < 4:
            outline = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    else:
        outline = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    bbox = {"x_min": int(x), "y_min": int(y), "x_max": int(x + w), "y_max": int(y + h)}
    return bbox, outline, contour_count


def _infer_openings(binary_inv: np.ndarray, bbox: dict[str, int]) -> list[dict[str, Any]]:
    openings: list[dict[str, Any]] = []
    x_min = int(bbox["x_min"])
    x_max = int(bbox["x_max"])
    y_min = int(bbox["y_min"])
    y_max = int(bbox["y_max"])
    cropped = binary_inv[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return openings

    h, w = cropped.shape
    boundaries = {
        "north": cropped[0 : min(5, h), :].mean(axis=0),
        "south": cropped[max(0, h - 5) : h, :].mean(axis=0),
        "west": cropped[:, 0 : min(5, w)].mean(axis=1),
        "east": cropped[:, max(0, w - 5) : w].mean(axis=1),
    }
    for side, arr in boundaries.items():
        zero_idx = np.where(arr < 20)[0]
        if len(zero_idx) == 0:
            continue
        start = int(zero_idx[0])
        prev = int(zero_idx[0])
        for idx in zero_idx[1:]:
            idx = int(idx)
            if idx - prev > 8:
                if prev - start >= 16:
                    openings.append(
                        {
                            "side": side,
                            "start_px": float(start + (x_min if side in {"north", "south"} else y_min)),
                            "end_px": float(prev + (x_min if side in {"north", "south"} else y_min)),
                            "width_px": float(prev - start),
                        }
                    )
                start = idx
            prev = idx
        if prev - start >= 16:
            openings.append(
                {
                    "side": side,
                    "start_px": float(start + (x_min if side in {"north", "south"} else y_min)),
                    "end_px": float(prev + (x_min if side in {"north", "south"} else y_min)),
                    "width_px": float(prev - start),
                }
            )
    return openings


def extract_cv_geometry(image: Image.Image) -> dict[str, Any] | None:
    if cv2 is None:
        return None

    gray = np.asarray(image.convert("L"))
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    bbox, outline, contour_count = _outline_from_contours(cleaned, image.width, image.height)
    width = max(1, int(bbox["x_max"] - bbox["x_min"]))
    height = max(1, int(bbox["y_max"] - bbox["y_min"]))
    min_length = max(24.0, min(width, height) * 0.22)
    lines = cv2.HoughLinesP(
        cleaned,
        1,
        np.pi / 180,
        threshold=max(28, int(min(width, height) * 0.15)),
        minLineLength=int(min_length),
        maxLineGap=12,
    )
    vertical_lines, horizontal_lines = _segments_to_lines(lines, min_length_px=min_length)
    openings = _infer_openings(cleaned, bbox)

    cropped = cleaned[bbox["y_min"] : bbox["y_max"], bbox["x_min"] : bbox["x_max"]]
    dark_pixel_ratio = float(cropped.mean() / 255.0) if cropped.size else 0.0
    return {
        "bbox_px": bbox,
        "outline_px": outline,
        "major_vertical_lines": vertical_lines,
        "major_horizontal_lines": horizontal_lines,
        "openings": openings,
        "dark_pixel_ratio": round(dark_pixel_ratio, 4),
        "parser": "opencv",
        "parser_metadata": {
            "cv_available": True,
            "contour_count": contour_count,
            "line_segment_count": 0 if lines is None else int(len(lines)),
            "binary_thresholding": "otsu+close",
        },
    }
