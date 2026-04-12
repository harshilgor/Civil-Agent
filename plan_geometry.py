"""
Deterministic geometry extraction for clean floor plans.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from plan_parser_cv import extract_cv_geometry


def _group_indices(indices: np.ndarray, min_gap: int = 4) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    groups = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx - prev > min_gap:
            groups.append((start, prev))
            start = idx
        prev = idx
    groups.append((start, prev))
    return groups


def _line_objects(groups: list[tuple[int, int]], orientation: str, start_px: int, end_px: int) -> list[dict[str, Any]]:
    lines = []
    for a, b in groups:
        center = (a + b) / 2.0
        lines.append(
            {
                "orientation": orientation,
                "position_px": round(center, 1),
                "thickness_px": round(b - a + 1, 1),
                "start_px": float(start_px),
                "end_px": float(end_px),
            }
        )
    return lines


def extract_plan_geometry(image: Image.Image, darkness_threshold: int = 205) -> dict[str, Any]:
    """
    Extract bounding box and major horizontal/vertical lines from a clean plan image.
    """
    cv_geometry = extract_cv_geometry(image)
    if cv_geometry:
        return cv_geometry

    gray = image.convert("L")
    arr = np.asarray(gray)
    dark = arr < darkness_threshold

    ys, xs = np.where(dark)
    if len(xs) == 0 or len(ys) == 0:
        bbox = {"x_min": 0, "y_min": 0, "x_max": image.width, "y_max": image.height}
        return {
            "bbox_px": bbox,
            "outline_px": [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
            "major_vertical_lines": [],
            "major_horizontal_lines": [],
            "openings": [],
            "dark_pixel_ratio": 0.0,
        }

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    cropped = dark[y_min : y_max + 1, x_min : x_max + 1]
    row_density = cropped.mean(axis=1)
    col_density = cropped.mean(axis=0)

    row_idx = np.where(row_density > 0.20)[0]
    col_idx = np.where(col_density > 0.20)[0]

    horizontal_groups = _group_indices(row_idx)
    vertical_groups = _group_indices(col_idx)

    horizontal_lines = _line_objects(horizontal_groups, "horizontal", x_min, x_max)
    vertical_lines = _line_objects(vertical_groups, "vertical", y_min, y_max)

    openings = infer_openings(cropped, x_min, y_min)
    return {
        "bbox_px": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "outline_px": [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)],
        "major_vertical_lines": vertical_lines,
        "major_horizontal_lines": horizontal_lines,
        "openings": openings,
        "dark_pixel_ratio": round(float(cropped.mean()), 4),
        "parser": "projection_fallback",
        "parser_metadata": {
            "cv_available": False,
            "line_segment_count": len(vertical_lines) + len(horizontal_lines),
        },
    }


def infer_openings(cropped_dark: np.ndarray, x_offset: int, y_offset: int) -> list[dict[str, Any]]:
    openings = []
    if cropped_dark.size == 0:
        return openings

    h, w = cropped_dark.shape
    boundaries = {
        "north": cropped_dark[0: min(4, h), :].mean(axis=0),
        "south": cropped_dark[max(0, h - 4): h, :].mean(axis=0),
        "west": cropped_dark[:, 0: min(4, w)].mean(axis=1),
        "east": cropped_dark[:, max(0, w - 4): w].mean(axis=1),
    }

    for side, arr in boundaries.items():
        light_idx = np.where(arr < 0.08)[0]
        for start, end in _group_indices(light_idx, min_gap=6):
            if end - start < 14:
                continue
            openings.append(
                {
                    "side": side,
                    "start_px": float(start + (x_offset if side in {"north", "south"} else y_offset)),
                    "end_px": float(end + (x_offset if side in {"north", "south"} else y_offset)),
                    "width_px": float(end - start),
                }
            )
    return openings


def render_plan_overlay(
    base_image: Image.Image,
    geometry: dict[str, Any],
    support_lines: list[dict[str, Any]] | None = None,
    blocked_zones: list[dict[str, Any]] | None = None,
    spaces: list[dict[str, Any]] | None = None,
) -> Image.Image:
    """
    Draw an editable-style overlay preview for user confirmation.
    """
    img = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")

    bbox = geometry.get("bbox_px", {})
    if bbox:
        draw.rectangle(
            [(bbox["x_min"], bbox["y_min"]), (bbox["x_max"], bbox["y_max"])],
            outline=ImageColor.getrgb("#e74c3c"),
            width=4,
        )

    for line in geometry.get("major_vertical_lines", []):
        x = line["position_px"]
        draw.line(
            [(x, line["start_px"]), (x, line["end_px"])],
            fill=(65, 90, 119, 170),
            width=max(2, int(line["thickness_px"])),
        )

    for line in geometry.get("major_horizontal_lines", []):
        y = line["position_px"]
        draw.line(
            [(line["start_px"], y), (line["end_px"], y)],
            fill=(65, 90, 119, 170),
            width=max(2, int(line["thickness_px"])),
        )

    for support in support_lines or []:
        classification = support.get("classification", "")
        if classification == "exterior":
            color = (46, 204, 113, 230)
            width = 4
        elif classification == "probable_bearing":
            color = (39, 174, 96, 220)
            width = 4
        else:
            color = (26, 188, 156, 185)
            width = 3
        if support.get("orientation") == "vertical":
            x = support["position_px"]
            draw.line([(x, bbox["y_min"]), (x, bbox["y_max"])], fill=color, width=width)
        else:
            y = support["position_px"]
            draw.line([(bbox["x_min"], y), (bbox["x_max"], y)], fill=color, width=width)

    for zone in blocked_zones or []:
        rect = zone.get("rect_px")
        if rect:
            draw.rectangle(rect, fill=(231, 76, 60, 60), outline=(231, 76, 60, 180), width=2)

    for space in spaces or []:
        rect = space.get("rect_px")
        if rect:
            draw.rectangle(rect, outline=(241, 196, 15, 180), width=2)
            label = f"{space.get('name', space.get('type', 'Space'))}"
            draw.text((rect[0] + 4, rect[1] + 4), label, fill=(241, 196, 15, 255))

    return img


def _inside_blocked_zone(x_ft: float, y_ft: float, blocked_zones: list[dict[str, Any]]) -> bool:
    for zone in blocked_zones:
        rect = zone.get("rect_ft")
        if not rect:
            continue
        if rect[0] <= x_ft <= rect[2] and rect[1] <= y_ft <= rect[3]:
            return True
    return False


def _candidate_positions(
    support_model: dict[str, Any],
    orientation: str,
    *,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    source = support_model.get("support_candidates") or support_model.get("support_lines") or []
    result = []
    for item in source:
        if item.get("orientation") != orientation:
            continue
        if active_only and not item.get("active", True):
            continue
        result.append(item)
    return result


def _dedupe_positions(values: list[float], min_gap_ft: float = 2.0) -> list[float]:
    ordered = sorted(float(v) for v in values)
    deduped: list[float] = []
    for value in ordered:
        if not deduped or abs(deduped[-1] - value) >= min_gap_ft:
            deduped.append(value)
    return deduped


def _select_grid_lines(
    dimension_ft: float,
    desired_bays: int,
    candidates: list[dict[str, Any]],
    *,
    strategy: str,
) -> list[float]:
    target_lines = max(2, int(desired_bays) + 1)
    interior_needed = max(0, target_lines - 2)
    regularized = [dimension_ft * idx / max(target_lines - 1, 1) for idx in range(target_lines)]
    if interior_needed == 0:
        return [0.0, dimension_ft]

    def score_for(candidate: dict[str, Any]) -> float:
        base = float(candidate.get("score", 0.0))
        support_overlap = float(candidate.get("support_zone_overlap", 0))
        open_overlap = float(candidate.get("open_zone_overlap", 0))
        center_bias = abs(float(candidate["position_ft"]) - dimension_ft / 2.0) / max(dimension_ft / 2.0, 1e-6)
        if strategy == "bearing_wall":
            return base + 0.08 * support_overlap - 0.05 * open_overlap
        if strategy == "open_zone":
            return base - 0.18 * open_overlap - 0.10 * (1.0 - center_bias)
        return base - 0.05 * abs(float(candidate["position_ft"]) - regularized[min(len(regularized) - 2, max(1, round(float(candidate["position_ft"]) / max(dimension_ft, 1e-6) * (target_lines - 1))))])

    interior_candidates = [
        candidate
        for candidate in candidates
        if 1.0 < float(candidate["position_ft"]) < dimension_ft - 1.0 and candidate.get("classification") != "partition"
    ]
    interior_candidates.sort(key=lambda item: (-score_for(item), -float(item.get("score", 0.0))))

    chosen: list[float] = []
    min_gap_ft = max(6.0, dimension_ft / max(target_lines, 3) * 0.6)
    for candidate in interior_candidates:
        position_ft = float(candidate["position_ft"])
        if any(abs(existing - position_ft) < min_gap_ft for existing in chosen):
            continue
        chosen.append(position_ft)
        if len(chosen) >= interior_needed:
            break

    if len(chosen) < interior_needed:
        for value in regularized[1:-1]:
            if any(abs(existing - value) < min_gap_ft * 0.6 for existing in chosen):
                continue
            chosen.append(value)
            if len(chosen) >= interior_needed:
                break

    return _dedupe_positions([0.0, *chosen[:interior_needed], dimension_ft])


def render_plan_structure_alignment(
    base_image: Image.Image,
    geometry: dict[str, Any],
    building_graph: dict[str, Any],
    support_model: dict[str, Any],
    selected_scheme: dict[str, Any],
    recommendation: dict[str, Any],
) -> Image.Image:
    img = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")

    bbox = geometry["bbox_px"]
    length_ft = float(building_graph["length_ft"])
    width_ft = float(building_graph["width_ft"])
    scale_x = float(building_graph["scale_ft_per_px_x"])
    scale_y = float(building_graph["scale_ft_per_px_y"])
    blocked_zones = support_model.get("blocked_zones", [])

    for zone in blocked_zones:
        rect = zone.get("rect_px")
        if rect:
            draw.rectangle(rect, fill=(192, 57, 43, 42), outline=(192, 57, 43, 120), width=2)

    vertical_candidates = _candidate_positions(support_model, "vertical")
    horizontal_candidates = _candidate_positions(support_model, "horizontal")
    strategy = selected_scheme.get("alignment_mode", "regularized")
    x_lines_ft = _select_grid_lines(length_ft, int(recommendation["bays_x"]), vertical_candidates, strategy=strategy)
    y_lines_ft = _select_grid_lines(width_ft, int(recommendation["bays_y"]), horizontal_candidates, strategy=strategy)

    x_lines_px = [bbox["x_min"] + value / max(scale_x, 1e-6) for value in x_lines_ft]
    y_lines_px = [bbox["y_min"] + value / max(scale_y, 1e-6) for value in y_lines_ft]

    for x in x_lines_px:
        draw.line([(x, bbox["y_min"]), (x, bbox["y_max"])], fill=(52, 152, 219, 170), width=3)
    for y in y_lines_px:
        draw.line([(bbox["x_min"], y), (bbox["x_max"], y)], fill=(142, 68, 173, 155), width=3)

    beam_dir = recommendation.get("beam_dir", "x")
    beam_spacing_ft = float(recommendation.get("beam_spacing", max(width_ft / 4.0, 8.0)))

    if beam_dir == "x":
        y_values_ft = list(np.arange(beam_spacing_ft, width_ft, beam_spacing_ft))
        for y_ft in y_values_ft:
            if _inside_blocked_zone(length_ft / 2.0, y_ft, blocked_zones):
                continue
            y_px = bbox["y_min"] + y_ft / max(scale_y, 1e-6)
            for idx in range(len(x_lines_px) - 1):
                draw.line([(x_lines_px[idx], y_px), (x_lines_px[idx + 1], y_px)], fill=(241, 196, 15, 185), width=2)
    else:
        x_values_ft = list(np.arange(beam_spacing_ft, length_ft, beam_spacing_ft))
        for x_ft in x_values_ft:
            if _inside_blocked_zone(x_ft, width_ft / 2.0, blocked_zones):
                continue
            x_px = bbox["x_min"] + x_ft / max(scale_x, 1e-6)
            for idx in range(len(y_lines_px) - 1):
                draw.line([(x_px, y_lines_px[idx]), (x_px, y_lines_px[idx + 1])], fill=(241, 196, 15, 185), width=2)

    for x_ft, x_px in zip(x_lines_ft, x_lines_px):
        for y_ft, y_px in zip(y_lines_ft, y_lines_px):
            if _inside_blocked_zone(x_ft, y_ft, blocked_zones) and x_ft not in {0.0, length_ft} and y_ft not in {0.0, width_ft}:
                continue
            radius = 6
            draw.ellipse([(x_px - radius, y_px - radius), (x_px + radius, y_px + radius)], fill=(231, 76, 60, 235))

    label = f"{selected_scheme['scheme_name']}: {recommendation['candidate_id']} | {recommendation['beam']} / {recommendation['girder']}"
    draw.rectangle([(bbox["x_min"], max(0, bbox["y_min"] - 30)), (min(img.width, bbox["x_min"] + 520), bbox["y_min"] - 4)], fill=(13, 17, 23, 210))
    draw.text((bbox["x_min"] + 8, max(2, bbox["y_min"] - 26)), label, fill=(255, 255, 255, 255))
    return img
