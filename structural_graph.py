"""
NetworkX-backed structural relationship graph for parsed plans.
"""

from __future__ import annotations

from typing import Any

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency path
    nx = None


def networkx_available() -> bool:
    return nx is not None


def _line_overlaps_rect(rect_ft: list[float], orientation: str, position_ft: float) -> bool:
    x0, y0, x1, y1 = rect_ft
    if orientation == "vertical":
        return x0 <= position_ft <= x1
    return y0 <= position_ft <= y1


def build_plan_structural_graph(
    building_graph: dict[str, Any],
    support_model: dict[str, Any],
    geometry_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for space in building_graph.get("spaces", []):
        nodes.append(
            {
                "id": f"space:{space['id']}",
                "kind": "space",
                "name": space["name"],
                "space_type": space["type"],
                "confidence": space["confidence"],
            }
        )

    for idx, boundary in enumerate(building_graph.get("boundaries", []), start=1):
        boundary_id = boundary.get("id") or f"boundary:{idx}"
        nodes.append(
            {
                "id": boundary_id,
                "kind": "boundary",
                "orientation": boundary["orientation"],
                "position_ft": boundary["position_ft"],
                "classification": boundary["classification"],
                "support_score": boundary["support_score"],
            }
        )
        for space in building_graph.get("spaces", []):
            if _line_overlaps_rect(space["rect_ft"], boundary["orientation"], float(boundary["position_ft"])):
                edges.append(
                    {
                        "source": boundary_id,
                        "target": f"space:{space['id']}",
                        "relationship": "bounds_or_crosses",
                    }
                )

    for idx, support in enumerate(support_model.get("support_lines", []), start=1):
        support_id = f"support:{idx}"
        nodes.append(
            {
                "id": support_id,
                "kind": "support",
                "name": support["name"],
                "orientation": support["orientation"],
                "position_ft": support["position_ft"],
                "classification": support.get("classification", support.get("source", "support")),
            }
        )
        for space in building_graph.get("spaces", []):
            if _line_overlaps_rect(space["rect_ft"], support["orientation"], float(support["position_ft"])):
                edges.append(
                    {
                        "source": support_id,
                        "target": f"space:{space['id']}",
                        "relationship": "supports_or_passes",
                    }
                )

    if geometry_model:
        for annotation in geometry_model.get("annotations", []):
            annotation_id = f"annotation:{annotation['id']}"
            nodes.append(
                {
                    "id": annotation_id,
                    "kind": "annotation",
                    "text": annotation["text"],
                    "annotation_kind": annotation["kind"],
                }
            )

    graph_obj = None
    if nx is not None:
        graph_obj = nx.Graph()
        for node in nodes:
            node_id = node["id"]
            attrs = {key: value for key, value in node.items() if key != "id"}
            graph_obj.add_node(node_id, **attrs)
        for edge in edges:
            graph_obj.add_edge(edge["source"], edge["target"], relationship=edge["relationship"])

    return {
        "engine": "networkx" if nx is not None else "basic",
        "graph": graph_obj,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "space_nodes": sum(1 for item in nodes if item["kind"] == "space"),
            "support_nodes": sum(1 for item in nodes if item["kind"] == "support"),
            "boundary_nodes": sum(1 for item in nodes if item["kind"] == "boundary"),
        },
    }
