"""
Shapely-backed plan geometry model builder.
"""

from __future__ import annotations

from typing import Any

try:
    from shapely.geometry import LineString, Point, Polygon, box
except Exception:  # pragma: no cover - optional dependency path
    LineString = None
    Point = None
    Polygon = None
    box = None


def shapely_available() -> bool:
    return all(item is not None for item in (LineString, Point, Polygon, box))


def _space_rect_to_ft(zone: dict[str, Any], bbox: dict[str, Any], width_px: float, height_px: float, scale_x: float, scale_y: float) -> list[float]:
    x0 = bbox["x_min"] + zone["x"] * width_px
    y0 = bbox["y_min"] + zone["y"] * height_px
    x1 = x0 + zone["w"] * width_px
    y1 = y0 + zone["h"] * height_px
    return [
        round((x0 - bbox["x_min"]) * scale_x, 3),
        round((y0 - bbox["y_min"]) * scale_y, 3),
        round((x1 - bbox["x_min"]) * scale_x, 3),
        round((y1 - bbox["y_min"]) * scale_y, 3),
    ]


def build_geometry_model(
    geometry: dict[str, Any],
    semantics: dict[str, Any],
    *,
    length_ft: float,
    width_ft: float,
    ocr_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bbox = geometry["bbox_px"]
    width_px = max(1.0, float(bbox["x_max"] - bbox["x_min"]))
    height_px = max(1.0, float(bbox["y_max"] - bbox["y_min"]))
    scale_x = float(length_ft) / width_px
    scale_y = float(width_ft) / height_px

    spaces = []
    for idx, zone in enumerate(semantics.get("zones", []) or [], start=1):
        rect_ft = _space_rect_to_ft(zone, bbox, width_px, height_px, scale_x, scale_y)
        space_record = {
            "id": f"S{idx:02d}",
            "name": zone["name"],
            "type": zone["type"],
            "confidence": zone["confidence"],
            "rect_ft": rect_ft,
            "rect_px": [
                round(bbox["x_min"] + zone["x"] * width_px, 1),
                round(bbox["y_min"] + zone["y"] * height_px, 1),
                round(bbox["x_min"] + (zone["x"] + zone["w"]) * width_px, 1),
                round(bbox["y_min"] + (zone["y"] + zone["h"]) * height_px, 1),
            ],
        }
        if shapely_available():
            polygon = box(rect_ft[0], rect_ft[1], rect_ft[2], rect_ft[3])
            space_record["polygon_ft"] = polygon
            space_record["polygon_coords_ft"] = [(round(x, 3), round(y, 3)) for x, y in polygon.exterior.coords]
        spaces.append(space_record)

    wall_lines = []
    for orientation_key, lines in (("vertical", geometry.get("major_vertical_lines", [])), ("horizontal", geometry.get("major_horizontal_lines", []))):
        for idx, line in enumerate(lines, start=1):
            if orientation_key == "vertical":
                position_ft = round((float(line["position_px"]) - bbox["x_min"]) * scale_x, 3)
                start_ft = round((float(line["start_px"]) - bbox["y_min"]) * scale_y, 3)
                end_ft = round((float(line["end_px"]) - bbox["y_min"]) * scale_y, 3)
                line_coords = [(position_ft, start_ft), (position_ft, end_ft)]
            else:
                position_ft = round((float(line["position_px"]) - bbox["y_min"]) * scale_y, 3)
                start_ft = round((float(line["start_px"]) - bbox["x_min"]) * scale_x, 3)
                end_ft = round((float(line["end_px"]) - bbox["x_min"]) * scale_x, 3)
                line_coords = [(start_ft, position_ft), (end_ft, position_ft)]
            record = {
                "id": f"W{orientation_key[0].upper()}{idx:03d}",
                "orientation": orientation_key,
                "position_ft": position_ft,
                "line_coords_ft": line_coords,
                "thickness_px": float(line["thickness_px"]),
            }
            if shapely_available():
                line_obj = LineString(line_coords)
                record["line_ft"] = line_obj
                record["buffer_ft"] = line_obj.buffer(max(0.25, float(line["thickness_px"]) * max(scale_x, scale_y) * 0.5), cap_style=2)
                record["length_ft"] = round(float(line_obj.length), 3)
            else:
                record["length_ft"] = round(abs(end_ft - start_ft), 3)
            wall_lines.append(record)

    openings = []
    for idx, opening in enumerate(geometry.get("openings", []), start=1):
        record = {"id": f"O{idx:03d}", **opening}
        openings.append(record)

    annotations = []
    if ocr_evidence:
        for idx, label in enumerate(ocr_evidence.get("room_labels", []), start=1):
            bbox_px = label.get("bbox_px")
            record = {
                "id": f"A{idx:03d}",
                "text": label["label"],
                "kind": "room_label",
                "room_type": label["room_type"],
                "bbox_px": bbox_px,
            }
            if bbox_px:
                cx_px = (bbox_px[0] + bbox_px[2]) / 2.0
                cy_px = (bbox_px[1] + bbox_px[3]) / 2.0
                point_ft = (
                    round((cx_px - bbox["x_min"]) * scale_x, 3),
                    round((cy_px - bbox["y_min"]) * scale_y, 3),
                )
                record["point_ft"] = point_ft
                if shapely_available():
                    record["point_geometry"] = Point(point_ft)
            annotations.append(record)

    model: dict[str, Any] = {
        "engine": "shapely" if shapely_available() else "basic",
        "bbox_px": bbox,
        "length_ft": float(length_ft),
        "width_ft": float(width_ft),
        "scale_ft_per_px_x": round(scale_x, 6),
        "scale_ft_per_px_y": round(scale_y, 6),
        "spaces": spaces,
        "wall_lines": wall_lines,
        "openings": openings,
        "annotations": annotations,
        "outline_coords_ft": [(0.0, 0.0), (float(length_ft), 0.0), (float(length_ft), float(width_ft)), (0.0, float(width_ft))],
    }
    if shapely_available():
        model["outline_polygon_ft"] = box(0.0, 0.0, float(length_ft), float(width_ft))
    return model
