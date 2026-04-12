"""
Civil Agent — Streamlit web interface for structural beam sizing.
Run: streamlit run app.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

# Ensure project modules resolve when launched from any cwd
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from architectural_constraints import ZONE_TYPES, make_constraint_map_figure, sanitize_constraint_zones
from asce7_loads import (
    OCCUPANCY_DEFAULTS,
    auto_load_brief,
    get_governing_combination,
    get_occupancy_defaults,
    get_seismic_loads,
    get_snow_load,
    get_wind_loads,
    list_supported_cities,
)
from beam_physics import apply_lrfd, calculate_demands, check_beam_design
from beams_data import (
    BEAMS_DF,
    COLUMN_SECTIONS,
    get_beam_by_index,
    get_column_by_index,
    get_num_beams,
    get_num_columns,
)
from civil_agent import CivilAgent, ConversationalCivilAgent
from column_physics import check_column_design, find_lightest_passing_w14
from combinatorial_optimizer import CombinatorialOptimizer, make_pareto_chart
from concrete_design import run_rc_beam_checks
from connection_design import design_moment_connection
from corrections_logger import log_correction
from design_reviewer import review_design
from drawing_processor import get_page_thumbnails, image_to_base64, pdf_to_images
from parser import parse
from plan_geometry import render_plan_overlay, render_plan_structure_alignment
from plan_pipeline import (
    analyze_uploaded_plan,
    build_confirmed_plan_state,
    generate_plan_schemes,
)
from layout_generator import (
    PRIORITY_LABELS,
    SUPPORTED_CITIES as GENERATIVE_CITIES,
    build_brief,
    make_tradeoff_chart,
    parse_design_brief,
    run_generative_design,
)
from report import (
    generate_beam_report_pdf,
    generate_rc_beam_report_pdf,
    generate_optimization_report,
    generate_3d_optimizer_report,
)
from lateral_system import compare_systems
from section_recommender import predict_section
from structural_graph import build_plan_structural_graph
from framing_plan import make_framing_plan_figure
from visualization_3d import (
    generate_3d_frame_html,
    generate_building_data,
    member_schedule_rows,
)
import vision_extractor
from vision_extractor import extract_from_image, merge_extractions

# --- Page config & theme ----------------------------------------------------
st.set_page_config(
    page_title="Civil Agent",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0d1b2a"
STEEL = "#415a77"
ACCENT = "#778da9"
BG = "#f8f9fa"


def _secret_or_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""

st.markdown(
    f"""
    <style>
    /* Main area */
    .stApp {{
        background: linear-gradient(180deg, {BG} 0%, #ffffff 40%);
    }}
    /* Sidebar dark navy */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, #1b263b 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: #e0e1dd !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        font-weight: 500;
    }}
    /* Primary button */
    .stButton > button {{
        background-color: {STEEL} !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
    }}
    .stButton > button:hover {{
        background-color: #1d3557 !important;
    }}
    /* Metric cards */
    div[data-testid="stMetricValue"] {{
        color: {NAVY};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

CHECK_LABELS = [
    ("moment_ratio", "Moment"),
    ("shear_ratio", "Shear"),
    ("defl_ratio", "Deflection"),
    ("ltb_ratio", "LTB"),
]

RATIO_KEYS = ("moment_ratio", "shear_ratio", "defl_ratio", "ltb_ratio", "flange_ratio", "web_ratio")
SUPPORTED_CITIES = list_supported_cities()
OCCUPANCY_OPTIONS = list(OCCUPANCY_DEFAULTS.keys())


def _local_buckling_util(details: dict) -> float:
    return max(float(details.get("flange_ratio", 0)), float(details.get("web_ratio", 0)))


def _ratio_color(value: float) -> str:
    if value < 0.85:
        return "#2ecc71"
    if value <= 1.0:
        return "#f39c12"
    return "#e74c3c"


def _controlling_failure(details: dict) -> tuple[str, float]:
    """Name and value of worst ratio > 1.0, else worst overall."""
    labels = {
        "moment_ratio": "moment",
        "shear_ratio": "shear",
        "defl_ratio": "deflection",
        "ltb_ratio": "LTB",
        "flange_ratio": "local buckling (flange)",
        "web_ratio": "local buckling (web)",
    }
    numeric = {k: float(details[k]) for k in labels if k in details and isinstance(details[k], (int, float))}
    if not numeric:
        return "unknown", 0.0
    over = {k: v for k, v in numeric.items() if v > 1.0}
    if over:
        k = max(over, key=over.get)
        return labels[k], over[k]
    k = max(numeric, key=numeric.get)
    return labels[k], numeric[k]


@st.cache_data(show_spinner=False, ttl=86400)
def _calculate_3d_hazards(
    city: str,
    occupancy: str,
    num_floors: int,
    floor_height: float,
    bay_length: float,
    bay_width: float,
    bays_x: int,
    bays_y: int,
    dead_psf: float,
    live_psf: float,
) -> dict:
    footprint_area = bay_length * bay_width * bays_x * bays_y
    total_height = num_floors * floor_height
    effective_floor_weight_kips = footprint_area * (dead_psf + 0.25 * live_psf) / 1000.0

    wind = get_wind_loads(
        city,
        total_height,
        exposure="B",
        occupancy=occupancy,
        floor_height_ft=floor_height,
    )
    snow = get_snow_load(city)

    seismic_error = None
    try:
        seismic = get_seismic_loads(
            city,
            num_floors,
            effective_floor_weight_kips,
            site_class="D",
            occupancy=occupancy,
            floor_height_ft=floor_height,
        )
    except Exception as exc:
        seismic_error = str(exc)
        defaults = get_occupancy_defaults(occupancy)
        seismic = {
            "city": city,
            "site_class": "D",
            "risk_category": defaults["risk_category"],
            "importance_factor": defaults["importance_factor"],
            "response_system": "OMF",
            "R": 3.5,
            "ss": 0.0,
            "s1": 0.0,
            "sds": 0.0,
            "sd1": 0.0,
            "sdc": "",
            "base_shear_kips": 0.0,
            "effective_weight_kips": round(effective_floor_weight_kips * num_floors, 2),
            "floor_forces": [],
            "api_url": "",
        }

    seismic_equivalent_psf = (
        seismic["base_shear_kips"] * 1000.0 / footprint_area
        if footprint_area > 0
        else 0.0
    )
    governing = get_governing_combination(
        DL=dead_psf,
        LL=live_psf,
        W=wind["roof_pressure_psf"],
        E=seismic_equivalent_psf,
        S=snow["roof_snow_psf"],
    )

    return {
        "wind": wind,
        "snow": snow,
        "seismic": seismic,
        "seismic_equivalent_psf": round(seismic_equivalent_psf, 2),
        "governing": governing,
        "effective_floor_weight_kips": round(effective_floor_weight_kips, 2),
        "seismic_error": seismic_error,
    }


@st.cache_resource(show_spinner=False)
def load_civil_agent() -> CivilAgent:
    """Train once; cached for app lifetime."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return CivilAgent()


@st.cache_resource(show_spinner=False)
def load_conversational_agent() -> ConversationalCivilAgent:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return ConversationalCivilAgent()


def neighbor_alternatives(
    span_ft: float,
    dead_load: float,
    live_load: float,
    lb_use: float | None,
    point_load: float,
    defl_limit: int,
    best_action: int,
) -> list[tuple[str, str]]:
    """
    Returns lines like ('Next lighter', 'W18x40 — FAILS moment (1.08)').
    """
    lines: list[tuple[str, str]] = []

    def summarize(name: str, passes: bool, details: dict) -> str:
        if passes:
            ratios = [float(details[k]) for k in RATIO_KEYS if k in details]
            mx = max(ratios) if ratios else 0.0
            return f"{name} — PASSES (max util {mx:.2f})"
        what, val = _controlling_failure(details)
        return f"{name} — FAILS {what} ({val:.2f})"

    if best_action > 0:
        n, beam = get_beam_by_index(best_action - 1)
        p, _, _, d = check_beam_design(
            span_ft, dead_load, live_load, beam,
            Lb_ft=lb_use, point_load=point_load, defl_limit=defl_limit,
        )
        lines.append(("Next lighter option", summarize(n, p, d)))

    if best_action < get_num_beams() - 1:
        n, beam = get_beam_by_index(best_action + 1)
        p, _, _, d = check_beam_design(
            span_ft, dead_load, live_load, beam,
            Lb_ft=lb_use, point_load=point_load, defl_limit=defl_limit,
        )
        lines.append(("Next heavier option", summarize(n, p, d)))

    return lines


def find_best_action_index(beam_name: str) -> int:
    from beams_data import BEAMS_DF

    m = BEAMS_DF["name"] == beam_name
    if not m.any():
        return 0
    return int(BEAMS_DF.index[m][0])


def connection_utilization_chart(conn: dict) -> plt.Figure:
    """Horizontal utilization bars for shear tab checks."""
    labels = ["Bolt shear", "Bolt bearing", "Tab shear", "Weld"]
    keys = ["bolt_shear", "bolt_bearing", "tab_shear", "weld"]
    ch = conn["checks"]
    values = [float(ch[k]) for k in keys]
    colors = [_ratio_color(v) for v in values]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.55, edgecolor="#333", linewidth=0.3)
    ax.axvline(1.0, color="#c1121f", linestyle="--", linewidth=2, label="Limit = 1.0")
    ax.set_xlim(0, max(1.15, max(values) * 1.05))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Utilization (demand / capacity)")
    ax.set_title("Shear tab connection checks")
    ax.legend(loc="lower right", fontsize=8)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    return fig


def utilization_chart(details: dict) -> plt.Figure:
    labels = ["Moment", "Shear", "Deflection", "LTB", "Local buckl."]
    values = [
        float(details["moment_ratio"]),
        float(details["shear_ratio"]),
        float(details["defl_ratio"]),
        float(details["ltb_ratio"]),
        _local_buckling_util(details),
    ]
    colors = [_ratio_color(v) for v in values]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.55, edgecolor="#333", linewidth=0.3)
    ax.axvline(1.0, color="#c1121f", linestyle="--", linewidth=2, label="Limit = 1.0")
    ax.set_xlim(0, max(1.15, max(values) * 1.05))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Utilization (demand / capacity)")
    ax.set_title("Check utilization")
    ax.legend(loc="lower right", fontsize=8)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    return fig


# --- Column utilization chart helper ------------------------------------------

COL_CHECK_LABELS = [
    ("axial_ratio", "Axial"),
    ("pm_ratio", "P-M Interaction"),
]
COL_RATIO_KEYS = ("axial_ratio", "pm_ratio", "flange_ratio", "web_ratio")


def column_utilization_chart(details: dict) -> plt.Figure:
    labels = ["Axial", "P-M Interaction", "Local buckl. (fl)", "Local buckl. (web)"]
    values = [
        float(details["axial_ratio"]),
        float(details["pm_ratio"]),
        float(details["flange_ratio"]),
        float(details["web_ratio"]),
    ]
    colors = [_ratio_color(v) for v in values]
    fig, ax = plt.subplots(figsize=(8, 3.0))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.55, edgecolor="#333", linewidth=0.3)
    ax.axvline(1.0, color="#c1121f", linestyle="--", linewidth=2, label="Limit = 1.0")
    ax.set_xlim(0, max(1.15, max(values) * 1.05))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Utilization (demand / capacity)")
    ax.set_title("Column check utilization")
    ax.legend(loc="lower right", fontsize=8)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    return fig


def col_neighbor_alternatives(
    height_ft: float, Pu: float, Mu: float, K: float, best_action: int,
) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []

    def summarize(name, passes, details):
        if passes:
            ratios = [float(details[k]) for k in COL_RATIO_KEYS if k in details]
            mx = max(ratios) if ratios else 0.0
            return f"{name} -- PASSES (max util {mx:.2f})"
        numeric = {k: float(details[k]) for k in COL_RATIO_KEYS
                   if k in details and float(details[k]) > 1.0}
        if numeric:
            k = max(numeric, key=numeric.get)
            lbl = {"axial_ratio": "axial", "pm_ratio": "P-M",
                   "flange_ratio": "flange buckl.", "web_ratio": "web buckl."}
            return f"{name} -- FAILS {lbl.get(k, k)} ({numeric[k]:.2f})"
        return f"{name} -- FAILS"

    if best_action > 0:
        n, col = get_column_by_index(best_action - 1)
        p, _, _, d = check_column_design(height_ft, Pu, Mu, col, K)
        lines.append(("Next lighter option", summarize(n, p, d)))
    if best_action < get_num_columns() - 1:
        n, col = get_column_by_index(best_action + 1)
        p, _, _, d = check_column_design(height_ft, Pu, Mu, col, K)
        lines.append(("Next heavier option", summarize(n, p, d)))
    return lines


def find_column_action_index(column_name: str) -> int:
    """Index into COLUMN_SECTIONS for neighbor lookups."""
    sub = COLUMN_SECTIONS[COLUMN_SECTIONS["name"] == column_name]
    if sub.empty:
        return 0
    return int(sub.index[0])


def _members_dataframe(members: list[dict], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(members or [])
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df[columns]


def _clean_editor_records(df: pd.DataFrame, numeric_columns: set[str]) -> list[dict]:
    records = []
    for raw in df.to_dict("records"):
        record = {}
        for key, value in raw.items():
            if pd.isna(value):
                record[key] = None
            elif key in numeric_columns:
                try:
                    record[key] = float(value)
                except (TypeError, ValueError):
                    record[key] = None
            else:
                record[key] = value

        if any(record.get(key) not in (None, "") for key in ("mark", "section")):
            records.append(record)
    return records


def _constraint_editor_records(df: pd.DataFrame, *, length_ft: float, width_ft: float) -> list[dict]:
    records = []
    for idx, raw in enumerate(df.to_dict("records")):
        zone_type = str(raw.get("zone_type") or "").strip().lower()
        name = str(raw.get("name") or "").strip()
        if not zone_type and not name:
            continue
        try:
            records.append(
                {
                    "id": raw.get("id") or f"constraint_{idx + 1}",
                    "name": name or ZONE_TYPES.get(zone_type, f"Zone {idx + 1}"),
                    "zone_type": zone_type,
                    "x_ft": float(raw.get("x_ft") or 0.0),
                    "y_ft": float(raw.get("y_ft") or 0.0),
                    "width_ft": float(raw.get("width_ft") or 0.0),
                    "height_ft": float(raw.get("height_ft") or 0.0),
                    "note": str(raw.get("note") or "").strip(),
                }
            )
        except (TypeError, ValueError):
            continue

    return sanitize_constraint_zones(records, length_ft=length_ft, width_ft=width_ft)


def drawing_review_page() -> None:
    st.title("Drawing Review")
    st.markdown(
        "Upload structural drawings. Civil Agent will extract member sizes, "
        "let you correct them, and find potential optimization opportunities."
    )

    if not _secret_or_env("OPENAI_API_KEY"):
        st.warning(
            "OPENAI_API_KEY is not set in this Streamlit process. You can still edit "
            "previous/manual extraction data, but GPT-4o extraction will be unavailable "
            "until the environment variable is set and the app is restarted."
        )

    uploaded_file = st.file_uploader(
        "Upload structural drawing (PDF)",
        type=["pdf"],
        help="Upload a PDF of structural drawings. Schedule pages work best.",
    )

    if uploaded_file:
        safe_name = os.path.basename(uploaded_file.name)
        upload_dir = os.path.join("data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        temp_path = os.path.join(upload_dir, safe_name)

        with open(temp_path, "wb") as handle:
            handle.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Converting PDF..."):
                images = pdf_to_images(temp_path)
        except Exception as exc:
            st.error(
                "Could not convert the PDF to images. Make sure Poppler is installed "
                f"and its bin folder is on PATH. Details: {exc}"
            )
            return

        st.success(f"Loaded {len(images)} pages")

        st.subheader("Select pages to analyze")
        st.caption(
            "Schedule pages, framing plans, and column schedules usually give the best results. "
            "To control cost and latency, analyze up to the first 5 pages here."
        )

        thumbnails = get_page_thumbnails(images[:5], max_width=180)
        cols = st.columns(max(1, min(len(thumbnails), 5)))
        selected_pages: list[int] = []

        for i, thumb in enumerate(thumbnails):
            with cols[i % len(cols)]:
                st.image(thumb, caption=f"Page {i + 1}")
                if st.checkbox("Analyze", key=f"drawing_page_{i}"):
                    selected_pages.append(i)

        if len(images) > 5:
            st.caption(f"... and {len(images) - 5} more pages. Analyze the first 5 pages in this view.")

        can_extract = bool(_secret_or_env("OPENAI_API_KEY"))
        if selected_pages and st.button(
            "Extract Members with GPT-4o",
            type="primary",
            disabled=not can_extract,
        ):
            extractions = []
            progress = st.progress(0)
            status = st.empty()

            for idx, page_num in enumerate(selected_pages):
                status.text(f"Analyzing page {page_num + 1}...")
                b64 = image_to_base64(images[page_num])
                result = extract_from_image(b64)
                st.session_state["vision_debug"] = dict(vision_extractor.LAST_DEBUG)
                extractions.append(result)
                progress.progress((idx + 1) / len(selected_pages))

            merged = merge_extractions(extractions)
            st.session_state["extracted"] = merged
            st.session_state["drawing_name"] = uploaded_file.name
            status.text("Extraction complete.")

        if selected_pages and not can_extract:
            st.info("Set OPENAI_API_KEY and restart the app to enable GPT-4o extraction.")

    if "extracted" not in st.session_state:
        st.info("Upload a PDF and run extraction to review members.")
        return

    extracted = st.session_state["extracted"]

    if "vision_debug" in st.session_state:
        debug = st.session_state["vision_debug"]
        with st.expander("GPT-4o debug output", expanded=True):
            st.write("API key present:", debug.get("api_key_present"))
            st.write("Image size (base64 chars):", debug.get("image_size_chars"))
            if debug.get("error"):
                st.error(f"Extraction/parsing error: {debug['error']}")
            st.text_area(
                "GPT-4o raw response",
                value=debug.get("raw_response") or "",
                height=220,
                key="gpt4o_raw_response_display",
            )

    st.subheader("Review Extracted Members")
    st.caption("Edit incorrect values, add missing members, or delete wrong entries before running the physics review.")

    st.markdown("**Beams**")
    edited_beams = st.data_editor(
        _members_dataframe(
            extracted.get("beams", []),
            ["mark", "section", "span_ft", "spacing_ft", "Lb_ft", "composite", "confidence"],
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="beam_editor",
    )

    st.markdown("**Girders**")
    edited_girders = st.data_editor(
        _members_dataframe(
            extracted.get("girders", []),
            ["mark", "section", "span_ft", "spacing_ft", "Lb_ft", "confidence"],
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="girder_editor",
    )

    st.markdown("**Columns**")
    edited_cols = st.data_editor(
        _members_dataframe(
            extracted.get("columns", []),
            ["mark", "section", "height_ft", "unbraced_ft", "Pu_kips", "K_factor", "confidence"],
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="col_editor",
    )

    st.markdown("**Floor Loads**")
    load_col1, load_col2 = st.columns(2)
    loads = extracted.get("loads", {}) or {}
    with load_col1:
        dead_psf = st.number_input(
            "Dead load (psf)",
            value=float(loads.get("dead_psf") or 50),
            min_value=0.0,
            key="drawing_dead_psf",
        )
    with load_col2:
        live_psf = st.number_input(
            "Live load (psf)",
            value=float(loads.get("live_psf") or 80),
            min_value=0.0,
            key="drawing_live_psf",
        )

    notes = extracted.get("notes")
    if notes:
        with st.expander("Extraction notes"):
            if isinstance(notes, list):
                for note in notes:
                    st.write(note)
            else:
                st.write(notes)

    if st.button("Find Optimization Opportunities", type="primary"):
        corrected = {
            "beams": _clean_editor_records(edited_beams, {"span_ft", "spacing_ft", "Lb_ft"}),
            "girders": _clean_editor_records(edited_girders, {"span_ft", "spacing_ft", "Lb_ft"}),
            "columns": _clean_editor_records(
                edited_cols,
                {"height_ft", "unbraced_ft", "Pu_kips", "K_factor"},
            ),
            "loads": {
                "dead_psf": dead_psf,
                "live_psf": live_psf,
            },
        }

        log_correction(
            extracted,
            corrected,
            st.session_state.get("drawing_name", "unknown"),
        )

        with st.spinner("Running physics checks..."):
            st.session_state["review_results"] = review_design(corrected, corrected["loads"])

    if "review_results" not in st.session_state:
        return

    results = st.session_state["review_results"]
    summary = results["summary"]

    st.divider()
    st.subheader("Optimization Report")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Members reviewed", summary["total_members"])
    m2.metric("Over-designed", summary["over_designed"])
    m3.metric("Steel savings", f"{summary['total_weight_saved_lbs']:,.0f} lbs")
    m4.metric("Cost savings", f"${summary['total_cost_saved']:,.0f}")

    if summary["savings_percent"] > 0:
        st.success(
            f"This design uses about {summary['savings_percent']:.1f}% more steel than the "
            f"current simplified checks require. Optimizing could save roughly "
            f"${summary['total_cost_saved']:,.0f} in structural steel."
        )
    else:
        st.success("Design appears well optimized under the current simplified assumptions.")

    rec_data = []
    for member_type in ("beams", "girders"):
        for member in results.get(member_type, []):
            saved = float(member.get("weight_saved_per_ft", 0) or 0)
            span = float(member.get("span_ft", 0) or 0)
            rec_data.append(
                {
                    "Type": member_type[:-1].title(),
                    "Mark": member.get("mark"),
                    "Current": member.get("current_section"),
                    "Optimal": member.get("optimal_section"),
                    "Saved (lb/ft)": round(saved, 1),
                    "Total saved (lbs)": round(saved * span, 0),
                    "Status": member.get("status", "Optimal"),
                }
            )

    if rec_data:
        st.markdown("**Beam and Girder Recommendations**")
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

    if results.get("columns"):
        st.markdown("**Column Notes**")
        st.dataframe(pd.DataFrame(results["columns"]), use_container_width=True, hide_index=True)

    try:
        st.download_button(
            label="Download Optimization Report",
            data=generate_optimization_report(
                results,
                st.session_state.get("drawing_name", "drawing"),
            ),
            file_name="optimization_report.pdf",
            mime="application/pdf",
            key="dl_optimization_pdf",
        )
    except Exception as exc:
        st.caption(f"Optimization PDF export unavailable: {exc}")


def optimizer_3d_page() -> None:
    st.title("3D Building Optimizer")
    st.caption("Design your building, see it in 3D, and compare baseline versus optimized steel in one view.")

    top_col1, top_col2 = st.columns(2)
    with top_col1:
        city = st.selectbox("City", options=SUPPORTED_CITIES, index=SUPPORTED_CITIES.index("Chicago"), key="viz3d_city")
    with top_col2:
        occupancy = st.selectbox("Occupancy", options=OCCUPANCY_OPTIONS, key="viz3d_occupancy")

    auto_loads_enabled = st.checkbox(
        "Auto-calculate ASCE 7-22 loads from city and occupancy",
        value=True,
        key="viz3d_auto_loads",
    )

    occupancy_defaults = get_occupancy_defaults(occupancy)
    if st.session_state.get("viz3d_last_occupancy") != occupancy:
        st.session_state["viz3d_dead"] = occupancy_defaults["dead_psf"]
        st.session_state["viz3d_live"] = occupancy_defaults["live_psf"]
        st.session_state["viz3d_last_occupancy"] = occupancy

    col1, col2 = st.columns(2)
    with col1:
        num_floors = st.slider("Floors", 1, 20, 8, key="viz3d_floors")
        bay_length = st.number_input("Bay length (ft)", value=40.0, min_value=10.0, step=5.0, key="viz3d_bay_length")
        bay_width = st.number_input("Bay width (ft)", value=30.0, min_value=10.0, step=5.0, key="viz3d_bay_width")
        bays_x = st.slider("Bays (X direction)", 1, 6, 3, key="viz3d_bays_x")
        bays_y = st.slider("Bays (Y direction)", 1, 6, 2, key="viz3d_bays_y")

    with col2:
        floor_height = st.number_input("Floor height (ft)", value=14.0, min_value=8.0, step=1.0, key="viz3d_floor_height")
        composite = st.checkbox("Composite beams", value=True, key="viz3d_composite")
        if auto_loads_enabled:
            auto_brief = auto_load_brief(
                city=city,
                num_floors=num_floors,
                floor_height_ft=floor_height,
                bay_length_ft=bay_length,
                bay_width_ft=bay_width,
                occupancy=occupancy,
            )
            dead_psf = float(auto_brief["dead_psf"])
            live_psf = float(auto_brief["live_psf"])
            st.metric("Dead load", f"{dead_psf:.0f} psf")
            st.metric("Live load", f"{live_psf:.0f} psf")
        else:
            dead_psf = st.number_input("Dead load (psf)", value=50.0, min_value=5.0, step=5.0, key="viz3d_dead")
            live_psf = st.number_input("Live load (psf)", value=80.0, min_value=10.0, step=5.0, key="viz3d_live")

    hazards = _calculate_3d_hazards(
        city=city,
        occupancy=occupancy,
        num_floors=num_floors,
        floor_height=floor_height,
        bay_length=bay_length,
        bay_width=bay_width,
        bays_x=bays_x,
        bays_y=bays_y,
        dead_psf=dead_psf,
        live_psf=live_psf,
    )
    wind = hazards["wind"]
    snow = hazards["snow"]
    seismic = hazards["seismic"]
    st.info(
        "Wind: "
        f"{wind['roof_pressure_psf']:.1f} psf  "
        f"Seismic: V={seismic['base_shear_kips']:.0f} kips  "
        f"Snow: {snow['roof_snow_psf']:.1f} psf  "
        f"Governing combo: {hazards['governing']['governing_name']}"
    )
    st.caption(
        f"Basic wind speed {wind['basic_wind_speed_mph']:.0f} mph | "
        f"Exposure {wind['exposure']} | "
        f"Ss={seismic['ss']:.2f}g, S1={seismic['s1']:.2f}g, "
        f"SDS={seismic['sds']:.2f}g, SD1={seismic['sd1']:.2f}g | "
        f"Site Class {seismic['site_class']}"
    )
    st.caption(
        f"Seismic weight uses D + 0.25L: {hazards['effective_floor_weight_kips']:.1f} kips/floor. "
        f"Wind and snow use embedded city lookup tables for preliminary review."
    )
    if hazards["seismic_error"]:
        st.warning(f"USGS seismic lookup fell back to zero lateral load: {hazards['seismic_error']}")

    if st.button("Generate 3D Model", type="primary", key="viz3d_generate"):
        with st.spinner("Designing building..."):
            building_data = generate_building_data(
                num_floors=num_floors,
                floor_height=floor_height,
                bay_length=bay_length,
                bay_width=bay_width,
                bays_x=bays_x,
                bays_y=bays_y,
                dead_psf=dead_psf,
                live_psf=live_psf,
                composite=composite,
                hazards=hazards,
            )
            building_data["hazards"] = hazards
            building_data["city"] = city
            building_data["occupancy"] = occupancy
            st.session_state["building_data_3d"] = building_data
            st.session_state.pop("system_comparison_results", None)

    building_data = st.session_state.get("building_data_3d")
    if not building_data:
        st.info("Set the building geometry and loads, then click **Generate 3D Model**.")
        return

    st.markdown("### Load Flow Visualization")
    load_mode = st.radio(
        "Show load flow:",
        ["Off", "Gravity (static)", "Gravity (animated)", "Lateral (wind)"],
        horizontal=True,
        key="viz3d_load_mode",
    )
    st.caption(
        "Arrows show how loads travel from floor slabs through beams and girders into columns and foundations. "
        "Arrow thickness = load magnitude."
    )

    html = generate_3d_frame_html(building_data, width=1200, height=640, load_mode=load_mode)
    components.html(html, height=660)

    before_after = building_data.get("before_after", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total members", building_data["total_members"])
    m2.metric("Before", f"{before_after.get('before_lbs', 0):,.0f} lbs")
    m3.metric("After", f"{before_after.get('after_lbs', 0):,.0f} lbs")
    m4.metric(
        "Saved",
        f"{before_after.get('saved_lbs', 0):,.0f} lbs",
        delta=f"-{before_after.get('saved_pct', 0):.1f}%",
    )

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Cost savings", f"${before_after.get('cost_savings', 0):,.0f}")
    m6.metric("Composite beams", "Yes" if building_data.get("composite") else "No")
    m7.metric("Checks", "PASS" if building_data.get("passes") else "FAIL")
    m8.metric("Occupancy", building_data.get("occupancy", occupancy))

    hazard_summary = building_data.get("hazards")
    if hazard_summary:
        st.markdown("**ASCE 7 hazard summary**")
        hz1, hz2, hz3, hz4 = st.columns(4)
        hz1.metric("City", building_data.get("city", city))
        hz2.metric("Roof wind", f"{hazard_summary['wind']['roof_pressure_psf']:.1f} psf")
        hz3.metric("Base shear", f"{hazard_summary['seismic']['base_shear_kips']:.0f} kips")
        hz4.metric("Roof snow", f"{hazard_summary['snow']['roof_snow_psf']:.1f} psf")
        st.caption(
            f"Governing combo: {hazard_summary['governing']['governing_name']} = "
            f"{hazard_summary['governing']['governing_value']:.1f} psf equivalent"
        )

    lateral_summary = building_data.get("lateral", {})
    lateral_recommendation = lateral_summary.get("recommendation") or {}
    lateral_results = lateral_summary.get("results") or {}
    if lateral_recommendation:
        st.markdown("**Lateral system**")
        lt1, lt2, lt3, lt4 = st.columns(4)
        lt1.metric("System", lateral_recommendation.get("name", "N/A"))
        wind_case = lateral_results.get("cases", {}).get("wind", {})
        seismic_case = lateral_results.get("cases", {}).get("seismic", {})
        lt2.metric(
            "Wind drift",
            f"{wind_case.get('roof_drift_in', 0):.2f} in",
            delta=(
                f"allow {wind_case.get('allowable_roof_drift_in', 0):.2f} in"
                if wind_case
                else None
            ),
        )
        lt3.metric(
            "Seismic drift",
            f"{seismic_case.get('roof_drift_in', 0):.2f} in",
            delta=(
                f"allow {seismic_case.get('allowable_roof_drift_in', 0):.2f} in"
                if seismic_case
                else None
            ),
        )
        lt4.metric("Lateral checks", "PASS" if lateral_results.get("passes", True) else "FAIL")
        st.caption(lateral_recommendation.get("reason", ""))
        if lateral_recommendation.get("cost_note"):
            st.caption(lateral_recommendation["cost_note"])

        drift_rows = []
        for case_name, case_data in lateral_results.get("cases", {}).items():
            drift_rows.append(
                {
                    "Case": case_name.title(),
                    "Roof drift (in)": case_data.get("roof_drift_in", 0),
                    "Allowable (in)": case_data.get("allowable_roof_drift_in", 0),
                    "Passes": case_data.get("passes", False),
                }
            )
        if drift_rows:
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Structural System Selector")
    sys_col1, sys_col2 = st.columns(2)
    with sys_col1:
        open_perim = st.checkbox("Open perimeter required", value=False, key="system_open_perimeter")
    with sys_col2:
        max_braces = st.slider("Max brace bays", 1, 4, 2, key="system_max_braces")

    if st.button("Compare Lateral Systems", key="compare_lateral_systems"):
        comparison_hazards = building_data.get("hazards", hazards)
        comparison = compare_systems(
            building_data,
            comparison_hazards["wind"],
            comparison_hazards["seismic"],
            building_data.get("optimized", {}).get("column_by_story", []),
            {"open_perimeter": open_perim, "max_brace_bays": max_braces},
        )
        st.session_state["system_comparison_results"] = comparison

    comparison = st.session_state.get("system_comparison_results")
    if comparison:
        cols = st.columns(3)
        for col, key, label in zip(
            cols,
            ["moment_frame", "braced_frame", "shear_wall"],
            ["Moment Frame", "Braced Frame", "Shear Wall"],
        ):
            with col:
                data = comparison[key]
                is_rec = key == comparison["recommendation"]
                if is_rec:
                    st.success(f"Recommended\n{label}")
                else:
                    st.info(label)
                st.metric("Cost premium", f"${data['total_cost_premium']:,.0f}")
                drift_ratio = max(float(data.get("max_drift", 0.0)), 1e-6)
                st.metric("Max drift", f"H/{int(1.0 / drift_ratio):,.0f}")
                st.metric("Rentable loss", f"{data.get('rentable_lost_sqft', 0):,.0f} sf")

        st.caption(
            f"Governing lateral load: {comparison['governing_load'].title()} | "
            f"Wind V = {comparison['V_wind_kips']:.0f} kips | "
            f"Seismic V = {comparison['V_seismic_kips']:.0f} kips"
        )
        st.write(comparison["reasoning"])

    st.markdown("**Representative design summary**")
    column_by_story = building_data.get("optimized", {}).get("column_by_story", [])
    top_column = column_by_story[-1].get("name") if column_by_story else "NONE"
    mid_column = column_by_story[len(column_by_story) // 2].get("name") if column_by_story else "NONE"
    ground_column = column_by_story[0].get("name") if column_by_story else "NONE"
    summary_rows = [
        {
            "Member": "Beam",
            "Baseline": building_data["baseline"]["beams"].get("name"),
            "Optimized": building_data["optimized"]["beams"].get("name"),
            "Savings (lb/ft)": round(
                float(building_data["baseline"]["beams"].get("weight", 0))
                - float(building_data["optimized"]["beams"].get("weight", 0)),
                1,
            ),
        },
        {
            "Member": "Girder",
            "Baseline": building_data["baseline"]["girder"].get("name"),
            "Optimized": building_data["optimized"]["girder"].get("name"),
            "Savings (lb/ft)": round(
                float(building_data["baseline"]["girder"].get("weight", 0))
                - float(building_data["optimized"]["girder"].get("weight", 0)),
                1,
            ),
        },
        {
            "Member": "Column (top floor)",
            "Baseline": building_data["baseline"]["column"].get("name"),
            "Optimized": top_column,
            "Savings (lb/ft)": round(
                float(building_data["baseline"]["column"].get("weight", 0))
                - float(column_by_story[-1].get("weight", 0) if column_by_story else 0),
                1,
            ),
        },
        {
            "Member": "Column (mid floor)",
            "Baseline": building_data["baseline"]["column"].get("name"),
            "Optimized": mid_column,
            "Savings (lb/ft)": round(
                float(building_data["baseline"]["column"].get("weight", 0))
                - float(column_by_story[len(column_by_story) // 2].get("weight", 0) if column_by_story else 0),
                1,
            ),
        },
        {
            "Member": "Column (ground floor)",
            "Baseline": building_data["baseline"]["column"].get("name"),
            "Optimized": ground_column,
            "Savings (lb/ft)": round(
                float(building_data["baseline"]["column"].get("weight", 0))
                - float(column_by_story[0].get("weight", 0) if column_by_story else 0),
                1,
            ),
        },
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    member_rows = member_schedule_rows(building_data)
    member_df = pd.DataFrame(member_rows)
    screenshot_path = _ROOT / "data" / "visualization_3d_screenshot.png"
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            "Download Member Schedule",
            data=member_df.to_csv(index=False).encode("utf-8"),
            file_name="civil_agent_member_schedule.csv",
            mime="text/csv",
            use_container_width=True,
            key="viz3d_csv",
        )
    with export_col2:
        try:
            pdf_bytes = generate_3d_optimizer_report(
                building_data,
                member_rows,
                screenshot_path=str(screenshot_path) if screenshot_path.exists() else None,
            )
            st.download_button(
                "Download Report",
                data=pdf_bytes,
                file_name="civil_agent_3d_optimizer_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="viz3d_pdf",
            )
        except Exception as exc:
            st.caption(f"3D optimizer PDF export unavailable: {exc}")

    with st.expander("Member schedule preview"):
        st.dataframe(member_df, use_container_width=True, hide_index=True)


def generative_design_page() -> None:
    st.title("Generative Design")
    st.caption(
        "Describe your building. Civil Agent generates and evaluates multiple structural layouts automatically."
    )

    input_mode = st.radio(
        "Input method",
        ["Natural language", "Structured inputs"],
        horizontal=True,
        key="generative_input_mode",
    )

    brief = None
    if input_mode == "Natural language":
        default_text = st.session_state.get(
            "generative_prompt_text",
            "8-story office building, Chicago, 120 by 90 feet, open floor plan, minimize cost",
        )
        user_input = st.text_area(
            "Describe your building",
            value=default_text,
            placeholder="8-story office building, Chicago, 120 by 90 feet, open floor plan, minimize cost",
            height=110,
            key="generative_prompt_text",
        )
        if st.button("Parse Description", key="parse_generative_brief"):
            parsed = parse_design_brief(user_input)
            st.session_state["generative_brief"] = parsed

        brief = st.session_state.get("generative_brief")
        if brief:
            st.markdown("**Extracted brief**")
            st.json(brief)
            st.caption("Using the extracted brief below. You can edit values by switching to structured inputs if needed.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            length = st.number_input("Length (ft)", min_value=30.0, value=120.0, step=5.0, key="gen_length")
            width = st.number_input("Width (ft)", min_value=30.0, value=90.0, step=5.0, key="gen_width")
            floors = st.slider("Floors", 1, 30, 8, key="gen_floors")
            occupancy = st.selectbox(
                "Occupancy",
                ["Office", "Retail", "Warehouse", "Residential", "Hospital"],
                key="gen_occupancy",
            )
        with col2:
            city = st.selectbox("City", GENERATIVE_CITIES, index=GENERATIVE_CITIES.index("Chicago"), key="gen_city")
            max_span = st.number_input("Max span (ft)", min_value=20.0, value=40.0, step=5.0, key="gen_max_span")
            priority_label = st.selectbox(
                "Optimize for",
                list(PRIORITY_LABELS.keys()),
                index=list(PRIORITY_LABELS.keys()).index("Best balance"),
                key="gen_priority_label",
            )
            interior_cols = st.checkbox("Allow interior columns", value=True, key="gen_interior_cols")
        brief = build_brief(
            length_ft=length,
            width_ft=width,
            num_floors=floors,
            floor_height_ft=14.0,
            occupancy=occupancy.lower(),
            city=city,
            priority=PRIORITY_LABELS[priority_label],
            max_span_ft=max_span,
            allow_interior_cols=interior_cols,
            composite=True,
        )

    constraint_rows = []
    if brief:
        st.markdown("### Architectural Constraints")
        st.caption(
            "Use a simple 2D constraint map to reserve open zones, mark brace-free areas, or favor a core/support corridor."
        )
        default_constraints = st.session_state.get(
            "generative_constraint_rows",
            [
                {
                    "name": "Open zone",
                    "zone_type": "no_columns",
                    "x_ft": round(float(brief["length_ft"]) * 0.34, 1),
                    "y_ft": round(float(brief["width_ft"]) * 0.30, 1),
                    "width_ft": round(float(brief["length_ft"]) * 0.24, 1),
                    "height_ft": round(float(brief["width_ft"]) * 0.22, 1),
                    "note": "Keep this area open",
                }
            ],
        )
        edited_constraints = st.data_editor(
            pd.DataFrame(default_constraints),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "zone_type": st.column_config.SelectboxColumn(
                    "Zone type",
                    options=list(ZONE_TYPES.keys()),
                    required=True,
                )
            },
            key="generative_constraint_editor",
        )
        constraint_rows = _constraint_editor_records(
            edited_constraints,
            length_ft=float(brief["length_ft"]),
            width_ft=float(brief["width_ft"]),
        )
        st.session_state["generative_constraint_rows"] = constraint_rows
        brief["architectural_constraints"] = constraint_rows

        if constraint_rows:
            constraint_fig = make_constraint_map_figure(brief)
            st.pyplot(constraint_fig, clear_figure=True)
            plt.close(constraint_fig)
        else:
            st.info("No architectural constraints are active for this run.")

    if st.button("Generate Layouts", type="primary", key="run_generative_design"):
        if input_mode == "Natural language":
            prompt_text = st.session_state.get("generative_prompt_text", "")
            brief = parse_design_brief(prompt_text)
            st.session_state["generative_brief"] = brief
        brief["architectural_constraints"] = st.session_state.get("generative_constraint_rows", [])

        with st.spinner("Generating and evaluating layouts..."):
            results = run_generative_design(brief)
            st.session_state["generative_results"] = results
            st.session_state["generative_run_brief"] = brief

    results = st.session_state.get("generative_results")
    run_brief = st.session_state.get("generative_run_brief", brief)
    if not results:
        st.info("Choose a building description or structured inputs, then click **Generate Layouts**.")
        return

    st.success(
        f"Evaluated {results['num_candidates_tried']} layouts. "
        f"{results['num_candidates_passed']} passed all structural checks."
    )

    if run_brief:
        with st.expander("Design brief used"):
            st.json(run_brief)
        if run_brief.get("loads_auto_calculated"):
            st.info(
                "Loads auto-calculated from ASCE 7-22: "
                f"Dead {run_brief.get('dead_psf', 0):.0f} psf  |  "
                f"Live {run_brief.get('live_psf', 0):.0f} psf  |  "
                f"Wind {run_brief.get('wind_psf', 0):.1f} psf  |  "
                f"Seismic V {run_brief.get('seismic_V_kips', 0):.0f} kips  |  "
                f"Governing {run_brief.get('governing_combo', 'N/A')}"
            )
            if run_brief.get("seismic_error"):
                st.caption(f"USGS seismic lookup fallback used: {run_brief['seismic_error']}")

    st.markdown("### Civil Agent Recommendation")
    st.write(results["explanation"])

    recommended = results.get("recommended")
    if not recommended:
        return

    st.markdown("### Top 5 Layouts")
    top5_df = pd.DataFrame(results["top_5"])[
        [
            "candidate_id",
            "span_x",
            "span_y",
            "beam",
            "girder",
            "column",
            "num_columns",
            "total_steel_lbs",
            "floor_depth_in",
            "total_cost",
        ]
    ].copy()
    top5_df.columns = [
        "Grid",
        "Span X",
        "Span Y",
        "Beam",
        "Girder",
        "Column",
        "Columns",
        "Steel (lbs)",
        "Depth (in)",
        "Total Cost",
    ]
    st.dataframe(
        top5_df.style.apply(
            lambda row: ["background: #1a472a; color: white" if row.name == 0 else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    top_candidates = results["top_5"]
    active_candidate_id = st.selectbox(
        "Inspect a layout",
        options=[item["candidate_id"] for item in top_candidates],
        index=0,
        key="generative_active_candidate_id",
    )
    active_candidate = next(
        (item for item in top_candidates if item["candidate_id"] == active_candidate_id),
        recommended,
    )

    st.markdown("### Tradeoff Analysis")
    fig = make_tradeoff_chart(results["all_results"], recommended)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.markdown("### Floor Framing Sketch")
    sketch_fig = make_framing_plan_figure(
        active_candidate,
        run_brief,
        title=f"Framing Sketch - {active_candidate['candidate_id']}",
    )
    st.pyplot(sketch_fig, clear_figure=True)
    plt.close(sketch_fig)

    st.markdown("### 3D Model - Selected Layout")
    hazards = _calculate_3d_hazards(
        city=run_brief["city"],
        occupancy=run_brief["occupancy"],
        num_floors=int(run_brief["num_floors"]),
        floor_height=float(run_brief["floor_height_ft"]),
        bay_length=float(active_candidate["render_bay_length"]),
        bay_width=float(active_candidate["render_bay_width"]),
        bays_x=int(active_candidate["render_bays_x"]),
        bays_y=int(active_candidate["render_bays_y"]),
        dead_psf=float(run_brief["dead_psf"]),
        live_psf=float(run_brief["live_psf"]),
    )
    building_data = generate_building_data(
        num_floors=int(run_brief["num_floors"]),
        floor_height=float(run_brief["floor_height_ft"]),
        bay_length=float(active_candidate["render_bay_length"]),
        bay_width=float(active_candidate["render_bay_width"]),
        bays_x=int(active_candidate["render_bays_x"]),
        bays_y=int(active_candidate["render_bays_y"]),
        dead_psf=float(run_brief["dead_psf"]),
        live_psf=float(run_brief["live_psf"]),
        composite=True,
        beam_spacing_ft=float(active_candidate["beam_spacing"]),
        hazards=hazards,
    )
    html = generate_3d_frame_html(building_data, width=1200, height=640)
    components.html(html, height=660)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Selected grid", active_candidate["candidate_id"])
    g2.metric("Estimated cost", f"${active_candidate['total_cost']:,.0f}")
    g3.metric("Steel", f"{active_candidate['total_steel_lbs']:,.0f} lbs")
    g4.metric("Columns", int(active_candidate["num_columns"]))

    st.divider()
    st.subheader("Full Combinatorial Search")
    st.caption(
        "Search across ALL design variables simultaneously: layout, spacing, composite ratio, "
        "column family, framing direction, and member grouping."
    )

    objectives_expander = st.expander("Customize objectives")
    with objectives_expander:
        w_cost = st.slider("Cost weight", 0, 100, 40, key="combo_w_cost")
        w_steel = st.slider("Steel weight", 0, 100, 20, key="combo_w_steel")
        w_depth = st.slider("Depth weight", 0, 100, 15, key="combo_w_depth")
        w_cols = st.slider("Columns weight", 0, 100, 15, key="combo_w_cols")
        w_complex = st.slider("Simplicity", 0, 100, 10, key="combo_w_complex")

    if st.button("Run Full Optimization", type="primary", key="run_full_combo"):
        combo_brief = run_brief or brief
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(done, total):
            progress_bar.progress(min(1.0, done / max(total, 1)))
            status_text.text(f"Evaluating {done}/{total} candidates...")

        objectives = {
            "total_cost": w_cost / 100,
            "total_steel": w_steel / 100,
            "floor_depth": w_depth / 100,
            "num_columns": w_cols / 100,
            "complexity": w_complex / 100,
        }

        opt = CombinatorialOptimizer(combo_brief)
        combo_results = opt.run(
            objectives=objectives,
            max_workers=8,
            progress_callback=update_progress,
        )
        st.session_state["combo_results"] = combo_results
        st.session_state["combo_brief"] = combo_brief
        if "error" in combo_results:
            status_text.text(combo_results["error"])
        else:
            status_text.text(
                f"Complete! Evaluated {combo_results['n_tried']} candidates, "
                f"{combo_results['n_passed']} passed, in {combo_results['search_time_s']}s"
            )

    combo_results = st.session_state.get("combo_results")
    combo_brief = st.session_state.get("combo_brief", run_brief)
    if combo_results and "error" not in combo_results:
        alternatives = combo_results.get("design_alternatives", [])
        if alternatives:
            st.markdown("### Design Alternatives Dashboard")
            st.caption(
                "Compare viable schemes by cost, steel, drift, and constructability, then switch the active scheme "
                "to update the 3D model and explanation."
            )
            alt_rows = []
            for alt in alternatives:
                candidate = alt["candidate"]
                alt_rows.append(
                    {
                        "Scheme": alt["title"],
                        "Grid": candidate["candidate_id"],
                        "Cost ($)": f"{candidate['total_cost']:,.0f}",
                        "Steel (lb)": f"{candidate['total_steel']:,.0f}",
                        "Drift (ft)": f"{candidate['lateral_drift_ft']:.2f}",
                        "Depth (in)": f"{candidate['floor_depth']:.1f}",
                        "Unique sections": int(candidate["unique_sections"]),
                        "Constructability": f"{candidate['constructability_score']:.1f}",
                    }
                )
            st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)

            alt_keys = [alt["key"] for alt in alternatives]
            active_alt_key = st.radio(
                "Active scheme",
                options=alt_keys,
                index=0,
                format_func=lambda key: next(
                    (
                        f"{alt['title']} - {alt['candidate']['candidate_id']}"
                        for alt in alternatives
                        if alt["key"] == key
                    ),
                    key,
                ),
                horizontal=True,
                key="combo_active_alternative_key",
            )
            active_alt = next(alt for alt in alternatives if alt["key"] == active_alt_key)
        else:
            active_alt = {"title": "Recommended", "candidate": combo_results["recommended"], "narrative": combo_results["explanation"]}

        top10_df = pd.DataFrame(combo_results["top_10"])
        st.dataframe(
            top10_df[
                [
                    "rank",
                    "bays_x",
                    "bays_y",
                    "beam_name",
                    "girder_name",
                    "col_name",
                    "composite",
                    "total_steel",
                    "total_cost",
                    "lateral_drift_ft",
                    "floor_depth",
                    "num_columns",
                    "unique_sections",
                    "constructability_score",
                    "score",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        pareto_fig = make_pareto_chart(
            combo_results["all_results"],
            combo_results["pareto_front"],
            combo_results["recommended"],
            highlighted=alternatives,
        )
        st.pyplot(pareto_fig, clear_figure=True)
        plt.close(pareto_fig)

        st.markdown("### Recommendation")
        st.write(combo_results["explanation"])
        if active_alt:
            st.markdown("### Active Scheme Narrative")
            st.info(active_alt["narrative"])

        combo_rec = active_alt["candidate"] if active_alt else combo_results["recommended"]
        combo_hazards = _calculate_3d_hazards(
            city=combo_brief["city"],
            occupancy=combo_brief["occupancy"],
            num_floors=int(combo_brief["num_floors"]),
            floor_height=float(combo_brief["floor_height_ft"]),
            bay_length=float(combo_rec["render_bay_length"]),
            bay_width=float(combo_rec["render_bay_width"]),
            bays_x=int(combo_rec["render_bays_x"]),
            bays_y=int(combo_rec["render_bays_y"]),
            dead_psf=float(combo_brief["dead_psf"]),
            live_psf=float(combo_brief["live_psf"]),
        )
        combo_building_data = generate_building_data(
            num_floors=int(combo_brief["num_floors"]),
            floor_height=float(combo_brief["floor_height_ft"]),
            bay_length=float(combo_rec["render_bay_length"]),
            bay_width=float(combo_rec["render_bay_width"]),
            bays_x=int(combo_rec["render_bays_x"]),
            bays_y=int(combo_rec["render_bays_y"]),
            dead_psf=float(combo_brief["dead_psf"]),
            live_psf=float(combo_brief["live_psf"]),
            composite=True,
            beam_spacing_ft=float(combo_rec["beam_spacing"]),
            composite_ratio=float(combo_rec["composite"]),
            hazards=combo_hazards,
        )
        combo_sketch = make_framing_plan_figure(
            combo_rec,
            combo_brief,
            title=f"Framing Sketch - {combo_rec['candidate_id']}",
        )
        st.markdown("### Floor Framing Sketch")
        st.pyplot(combo_sketch, clear_figure=True)
        plt.close(combo_sketch)
        st.markdown(f"### 3D Model - {active_alt['title'] if active_alt else 'Selected Scheme'}")
        combo_html = generate_3d_frame_html(combo_building_data, width=1200, height=640)
        components.html(combo_html, height=660)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active grid", combo_rec["candidate_id"])
        c2.metric("Estimated cost", f"${combo_rec['total_cost']:,.0f}")
        c3.metric("Steel", f"{combo_rec['total_steel']:,.0f} lbs")
        c4.metric("Constructability", f"{combo_rec['constructability_score']:.1f}")


def plan_intelligence_page() -> None:
    st.title("Plan Intelligence")
    st.caption(
        "Upload a clean floor plan. Civil Agent converts it into structured building intelligence, "
        "lets you confirm key assumptions, then generates and evaluates preliminary framing schemes."
    )

    uploaded_file = st.file_uploader(
        "Upload floor plan image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        key="plan_intelligence_upload",
    )

    if uploaded_file and st.button("Analyze Floor Plan", type="primary", key="plan_analyze_btn"):
        with st.spinner("Ingesting, preprocessing, and interpreting the plan..."):
            analysis = analyze_uploaded_plan(uploaded_file)
            st.session_state["plan_analysis"] = analysis
            st.session_state.pop("plan_state", None)
            st.session_state.pop("plan_schemes", None)

    analysis = st.session_state.get("plan_analysis")
    if not analysis:
        st.info("Upload a clean residential plan and click **Analyze Floor Plan** to begin.")
        return

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Original plan**")
        st.image(analysis["document"]["selected_page"], use_container_width=True)
    with img_col2:
        st.markdown("**Preprocessed plan**")
        st.image(analysis["document"]["preprocessed_page"], use_container_width=True)

    st.markdown("### Scale Calibration")
    st.caption(
        "Scale is a formal confirmation step. Review the detected plan dimensions and correct them before any structural logic runs."
    )
    detected_scale = analysis["scale"]
    semantics = analysis["semantics"]
    ocr_evidence = analysis.get("ocr_evidence", {})
    text_artifacts = analysis.get("text_artifacts", {})
    scale_artifacts = analysis.get("scale_artifacts", {})

    cal1, cal2, cal3, cal4 = st.columns(4)
    with cal1:
        length_ft = st.number_input(
            "Overall length (ft)",
            min_value=10.0,
            value=float(detected_scale.get("length_ft") or 60.0),
            step=1.0,
            key="plan_length_ft",
        )
    with cal2:
        width_ft = st.number_input(
            "Overall width (ft)",
            min_value=10.0,
            value=float(detected_scale.get("width_ft") or 40.0),
            step=1.0,
            key="plan_width_ft",
        )
    with cal3:
        num_floors = st.slider("Floors", 1, 4, 1, key="plan_num_floors")
    with cal4:
        floor_height_ft = st.number_input("Floor height (ft)", min_value=8.0, value=10.0, step=1.0, key="plan_floor_height")

    cal5, cal6 = st.columns(2)
    with cal5:
        occupancy = st.selectbox(
            "Occupancy",
            ["Residential", "Office", "Retail"],
            index=0,
            key="plan_occupancy",
        )
    with cal6:
        city = st.selectbox(
            "City",
            GENERATIVE_CITIES,
            index=GENERATIVE_CITIES.index("Chicago"),
            key="plan_city",
        )

    st.caption(
        f"Detected dimension strings: {', '.join(semantics.get('detected_dimension_strings', [])[:6]) or 'none found'}"
    )
    if semantics.get("notes"):
        st.caption(f"Semantic notes: {semantics['notes']}")
    with st.expander("Parser evidence"):
        st.write(
            {
                "ingestion_engine": analysis["document"].get("ingestion_engine", "unknown"),
                "geometry_parser": analysis["geometry"].get("parser", "unknown"),
                "ocr_source": ocr_evidence.get("source", "unavailable"),
                "dimension_strings": ocr_evidence.get("dimension_strings", [])[:10],
                "room_labels": [item.get("label") for item in ocr_evidence.get("room_labels", [])[:10]],
                "text_summary": text_artifacts.get("summary", {}),
                "phrase_summary": text_artifacts.get("phrase_summary", {}),
                "best_scale_candidate": scale_artifacts.get("best_candidate", {}),
            }
        )

    if st.button("Confirm Scale and Build Overlay", key="plan_confirm_scale"):
        with st.spinner("Building internal geometry, zoning, and support inference..."):
            plan_state = build_confirmed_plan_state(
                analysis,
                length_ft=length_ft,
                width_ft=width_ft,
                num_floors=num_floors,
                floor_height_ft=floor_height_ft,
                occupancy=occupancy.lower(),
            )
            st.session_state["plan_state"] = plan_state
            st.session_state.pop("plan_schemes", None)

    plan_state = st.session_state.get("plan_state")
    if not plan_state:
        return

    st.markdown("### Confirm Overlay Assumptions")
    st.caption(
        "The overlay is intentionally selective: bright green lines are the few support opportunities the engine currently trusts most, yellow boxes are interpreted spaces, and red overlays mark support-avoidance zones."
    )
    st.image(plan_state["overlay_image"], use_container_width=True)

    spaces_df = pd.DataFrame(plan_state["building_graph"]["spaces"])
    support_candidates = plan_state["support_model"].get("support_candidates", plan_state["support_model"]["support_lines"])
    supports_df = pd.DataFrame(support_candidates).copy()
    if not supports_df.empty and "active" not in supports_df.columns:
        supports_df["active"] = True
    boundaries_df = pd.DataFrame(plan_state["building_graph"].get("boundaries", []))

    st.markdown("**Detected spaces**")
    edited_spaces = st.data_editor(
        spaces_df[["name", "type", "confidence", "rect_ft"]],
        num_rows="dynamic",
        use_container_width=True,
        key="plan_spaces_editor",
    )

    support_col1, support_col2 = st.columns([1.35, 1.0])
    with support_col1:
        st.markdown("**Ranked support candidates**")
        support_columns = [column for column in ["active", "name", "classification", "orientation", "position_ft", "score", "source", "reasons"] if column in supports_df.columns]
        edited_supports = st.data_editor(
            supports_df[support_columns],
            num_rows="dynamic",
            use_container_width=True,
            key="plan_supports_editor",
        )
    with support_col2:
        st.markdown("**Wall classification summary**")
        if not boundaries_df.empty:
            wall_summary = (
                boundaries_df.groupby(["classification", "orientation"], dropna=False)
                .size()
                .reset_index(name="Count")
                .sort_values(["classification", "orientation"])
            )
            st.dataframe(wall_summary, use_container_width=True, hide_index=True)
            with st.expander("Detailed wall evidence"):
                boundary_columns = [
                    column
                    for column in ["orientation", "position_ft", "length_ft", "thickness_px", "classification", "support_score", "reasons"]
                    if column in boundaries_df.columns
                ]
                st.dataframe(boundaries_df[boundary_columns], use_container_width=True, hide_index=True)
        else:
            st.info("No wall classification data is available for this plan yet.")

    if not supports_df.empty:
        st.caption(
            "Use the `active` toggle to promote or demote inferred support opportunities before scheme generation. Lower-scoring partitions stay visible here for transparency but are not activated by default."
        )


    if st.button("Apply Overlay Edits", key="plan_apply_overlay"):
        graph = plan_state["building_graph"]
        bbox = graph["bbox_px"]
        scale_x = float(graph["scale_ft_per_px_x"])
        scale_y = float(graph["scale_ft_per_px_y"])

        updated_spaces = []
        original_spaces = graph["spaces"]
        for idx, row in edited_spaces.iterrows():
            base = original_spaces[idx] if idx < len(original_spaces) else None
            rect_ft = row.get("rect_ft", base["rect_ft"] if base else [0, 0, 5, 5])
            if not isinstance(rect_ft, list):
                rect_ft = list(rect_ft)
            rect_px = [
                bbox["x_min"] + rect_ft[0] / max(scale_x, 1e-6),
                bbox["y_min"] + rect_ft[1] / max(scale_y, 1e-6),
                bbox["x_min"] + rect_ft[2] / max(scale_x, 1e-6),
                bbox["y_min"] + rect_ft[3] / max(scale_y, 1e-6),
            ]
            updated_spaces.append(
                {
                    **(base or {}),
                    "name": row["name"],
                    "type": str(row["type"]).lower(),
                    "confidence": row["confidence"],
                    "rect_ft": rect_ft,
                    "rect_px": rect_px,
                    "open_zone_candidate": str(row["type"]).lower() in {"garage", "living", "family", "great room", "patio", "porch", "entry"},
                    "support_friendly": str(row["type"]).lower() in {"bedroom", "bath", "corridor", "closet", "utility", "laundry", "study", "hall"},
                }
              )
        graph["spaces"] = updated_spaces

        updated_candidates = []
        active_supports = []
        for _, row in edited_supports.iterrows():
            if not bool(row.get("active", True)):
                is_active = False
            else:
                is_active = True
            position_ft = float(row["position_ft"])
            position_px = (
                bbox["x_min"] + position_ft / max(scale_x, 1e-6)
                if row["orientation"] == "vertical"
                else bbox["y_min"] + position_ft / max(scale_y, 1e-6)
            )
            candidate_record = {
                "name": row["name"],
                "orientation": row["orientation"],
                "position_ft": position_ft,
                "position_px": position_px,
                "source": row["source"],
                "classification": row.get("classification", "possible_support"),
                "score": float(row.get("score", 0.5)),
                "reasons": list(row.get("reasons", [])) if isinstance(row.get("reasons", []), list) else [str(row.get("reasons"))],
                "active": is_active,
            }
            updated_candidates.append(candidate_record)
            if is_active:
                active_supports.append(
                    {
                        "name": candidate_record["name"],
                        "orientation": candidate_record["orientation"],
                        "position_ft": candidate_record["position_ft"],
                        "position_px": candidate_record["position_px"],
                        "source": candidate_record["source"],
                        "classification": candidate_record["classification"],
                        "score": candidate_record["score"],
                        "reasons": candidate_record["reasons"],
                    }
                )
        plan_state["support_model"]["support_candidates"] = updated_candidates
        plan_state["support_model"]["support_lines"] = active_supports
        blocked_zones = [
            {"name": space["name"], "type": space["type"], "rect_px": space["rect_px"], "rect_ft": space["rect_ft"]}
            for space in updated_spaces
            if space["open_zone_candidate"]
        ]
        plan_state["support_model"]["blocked_zones"] = blocked_zones
        plan_state["structural_graph"] = build_plan_structural_graph(
            plan_state["building_graph"],
            plan_state["support_model"],
            plan_state.get("geometry_model"),
        )
        plan_state["overlay_image"] = render_plan_overlay(
            plan_state["document"]["selected_page"],
            plan_state["geometry"],
            support_lines=active_supports,
            blocked_zones=blocked_zones,
            spaces=updated_spaces,
        )
        st.session_state["plan_state"] = plan_state
        st.rerun()

    if st.button("Generate Preliminary Framing Schemes", type="primary", key="plan_generate_schemes"):
        with st.spinner("Generating and evaluating multiple structural concepts..."):
            schemes = generate_plan_schemes(plan_state, city=city)
            st.session_state["plan_schemes"] = schemes

    schemes = st.session_state.get("plan_schemes")
    if not schemes:
        return

    if plan_state.get("structural_graph"):
        graph_summary = plan_state["structural_graph"]["summary"]
        st.caption(
            f"Structural graph foundation: {graph_summary['node_count']} nodes, {graph_summary['edge_count']} edges, "
            f"{graph_summary['support_nodes']} support nodes."
        )
    if plan_state.get("semantic_artifacts"):
        semantic_summary = plan_state["semantic_artifacts"]["summary"]
        st.caption(
            f"Deterministic zoning layer: {semantic_summary['zone_count']} zones, "
            f"{semantic_summary['open_zone_count']} open-zone candidates, "
            f"{semantic_summary['support_friendly_count']} support-friendly zones."
        )

    st.markdown("### Candidate Structural Schemes")
    scheme_rows = []
    for scheme in schemes:
        rec = scheme["recommended"]
        scheme_rows.append(
            {
                "Scheme": scheme["scheme_name"],
                "Description": scheme["description"],
                "Grid": rec["candidate_id"],
                "Beam": rec["beam"],
                "Girder": rec["girder"],
                "Column": rec["column"],
                "Cost": rec["total_cost"],
                "Steel (lb)": rec["total_steel_lbs"],
                "Depth (in)": rec["floor_depth_in"],
                "Columns": rec["num_columns"],
            }
        )
    scheme_df = pd.DataFrame(scheme_rows)
    st.dataframe(scheme_df, use_container_width=True, hide_index=True)

    scheme_names = [scheme["scheme_name"] for scheme in schemes]
    selected_name = st.selectbox("Inspect scheme", scheme_names, key="plan_selected_scheme")
    selected_scheme = next(s for s in schemes if s["scheme_name"] == selected_name)
    selected_result = selected_scheme["result"]
    selected_rec = selected_scheme["recommended"]
    selected_brief = selected_scheme["brief"]

    st.markdown("### Structural Explanation")
    st.write(selected_result["explanation"])
    st.caption(selected_scheme.get("structural_story", ""))

    st.markdown("### Plan-To-Structure Alignment")
    st.caption(
        "This view keeps the architecture and structural scheme on the same page: blue and purple lines show the primary framing grid, yellow lines show beam runs, and red nodes show column stack opportunities."
    )
    alignment_image = render_plan_structure_alignment(
        plan_state["document"]["selected_page"],
        plan_state["geometry"],
        plan_state["building_graph"],
        plan_state["support_model"],
        selected_scheme,
        selected_rec,
    )
    align_col1, align_col2 = st.columns(2)
    with align_col1:
        st.markdown("**Original plan**")
        st.image(plan_state["document"]["selected_page"], use_container_width=True)
    with align_col2:
        st.markdown("**Selected scheme aligned to plan**")
        st.image(alignment_image, use_container_width=True)

    load_mode = st.radio(
        "3D load mode",
        ["Off", "Gravity (static)", "Gravity (animated)", "Lateral (wind)"],
        horizontal=True,
        key="plan_load_mode",
    )

    hazards = _calculate_3d_hazards(
        city=city,
        occupancy=selected_brief["occupancy"],
        num_floors=int(selected_brief["num_floors"]),
        floor_height=float(selected_brief["floor_height_ft"]),
        bay_length=float(selected_rec["render_bay_length"]),
        bay_width=float(selected_rec["render_bay_width"]),
        bays_x=int(selected_rec["render_bays_x"]),
        bays_y=int(selected_rec["render_bays_y"]),
        dead_psf=float(selected_brief["dead_psf"]),
        live_psf=float(selected_brief["live_psf"]),
    )
    building_data = generate_building_data(
        num_floors=int(selected_brief["num_floors"]),
        floor_height=float(selected_brief["floor_height_ft"]),
        bay_length=float(selected_rec["render_bay_length"]),
        bay_width=float(selected_rec["render_bay_width"]),
        bays_x=int(selected_rec["render_bays_x"]),
        bays_y=int(selected_rec["render_bays_y"]),
        dead_psf=float(selected_brief["dead_psf"]),
        live_psf=float(selected_brief["live_psf"]),
        composite=bool(selected_brief["composite"]),
        beam_spacing_ft=float(selected_rec["beam_spacing"]),
        hazards=hazards,
    )
    html = generate_3d_frame_html(building_data, width=1200, height=640, load_mode=load_mode)
    components.html(html, height=660)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scheme", selected_scheme["scheme_name"])
    m2.metric("Estimated cost", f"${selected_rec['total_cost']:,.0f}")
    m3.metric("Steel", f"{selected_rec['total_steel_lbs']:,.0f} lbs")
    m4.metric("Columns", int(selected_rec["num_columns"]))

    member_rows = member_schedule_rows(building_data)
    member_df = pd.DataFrame(member_rows)
    if not member_df.empty:
        grouped = (
            member_df.groupby(["Type", "Optimized Section"], dropna=False)
            .agg(Quantity=("ID", "count"), Total_Length_ft=("Length (ft)", "sum"))
            .reset_index()
        )
        st.markdown("### Member Schedule Summary")
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        with st.expander("Full member schedule preview"):
            st.dataframe(member_df, use_container_width=True, hide_index=True)


def concrete_beam_page() -> None:
    st.title("RC beam (ACI 318)")
    st.caption(
        "Check a rectangular reinforced-concrete beam using the standalone ACI 318-19 engine. "
        "This keeps the concrete code-check path separate from the steel sizing workflows."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        b_in = st.number_input("Width b (in)", min_value=6.0, value=14.0, step=1.0)
        h_in = st.number_input("Total depth h (in)", min_value=8.0, value=24.0, step=1.0)
        d_in = st.number_input("Effective depth d (in)", min_value=4.0, value=21.5, step=0.5)
    with c2:
        fc_psi = st.number_input("Concrete strength f'c (psi)", min_value=2500.0, value=4000.0, step=500.0)
        fy_psi = st.number_input("Steel yield fy (psi)", min_value=40000.0, value=60000.0, step=5000.0)
        tension_bar_count = st.number_input("Bottom bar count", min_value=1, max_value=12, value=3, step=1)
    with c3:
        tension_bar_size = st.selectbox("Bottom bar size", [3, 4, 5, 6, 7, 8, 9, 10, 11], index=6)
        Mu_kip_ft = st.number_input("Factored moment Mu (kip-ft)", min_value=0.0, value=200.0, step=10.0)
        Vu_kips = st.number_input("Factored shear Vu (kips)", min_value=0.0, value=20.0, step=1.0)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        stirrup_bar_size = st.selectbox("Stirrup size", [0, 3, 4, 5], index=1, help="Use 0 for no stirrups.")
    with s2:
        stirrup_spacing_in = st.number_input("Stirrup spacing (in)", min_value=0.0, value=8.0, step=1.0)
    with s3:
        stirrup_legs = st.selectbox("Stirrup legs", [2, 4], index=0)
    with s4:
        use_detailed_shear = st.checkbox("Detailed shear Vc", value=False)

    if not st.button("Run ACI 318 Checks", type="primary", key="run_rc_beam_checks"):
        st.info("Enter beam properties and click **Run ACI 318 Checks** to evaluate flexure and shear.")
        return

    result = run_rc_beam_checks(
        b_in=b_in,
        h_in=h_in,
        d_in=d_in,
        fc_psi=fc_psi,
        fy_psi=fy_psi,
        tension_bar_count=int(tension_bar_count),
        tension_bar_size=int(tension_bar_size),
        stirrup_bar_size=int(stirrup_bar_size),
        stirrup_spacing_in=float(stirrup_spacing_in),
        stirrup_legs=int(stirrup_legs),
        Mu_kip_ft=Mu_kip_ft,
        Vu_kips=Vu_kips,
        use_detailed_shear=use_detailed_shear,
    )

    flexure = result["flexure"]
    shear = result["shear"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall status", result["overall_status"])
    m2.metric("Controlling check", result["controlling_check"].title())
    m3.metric("Controlling DCR", f"{result['controlling_dcr']:.3f}")

    cflex, cshear = st.columns(2)
    with cflex:
        st.markdown("### Flexure")
        st.metric("DCR", f"{flexure.dcr:.3f}", help=flexure.summary())
        st.metric("Capacity", f"{flexure.capacity:,.0f} lb-in")
        st.metric("Demand", f"{flexure.demand:,.0f} lb-in")
        st.caption(f"{flexure.status.value.upper()} | {flexure.code_clause}")
        if flexure.warnings:
            for warning in flexure.warnings:
                st.warning(warning)
        with st.expander("Flexure calculation steps"):
            flexure_steps = pd.DataFrame([step.to_dict() for step in flexure.calc_steps])
            st.dataframe(flexure_steps, use_container_width=True, hide_index=True)

    with cshear:
        st.markdown("### Shear")
        st.metric("DCR", f"{shear.dcr:.3f}", help=shear.summary())
        st.metric("Capacity", f"{shear.capacity:,.0f} lb")
        st.metric("Demand", f"{shear.demand:,.0f} lb")
        st.caption(f"{shear.status.value.upper()} | {shear.code_clause}")
        if shear.warnings:
            for warning in shear.warnings:
                st.warning(warning)
        with st.expander("Shear calculation steps"):
            shear_steps = pd.DataFrame([step.to_dict() for step in shear.calc_steps])
            st.dataframe(shear_steps, use_container_width=True, hide_index=True)

    report_bytes = generate_rc_beam_report_pdf(result)
    st.download_button(
        "Download RC Beam Report",
        data=report_bytes,
        file_name="rc_beam_report.pdf",
        mime="application/pdf",
    )


def conversational_chat_page() -> None:
    st.title("Chat with Civil Agent")
    st.caption(
        "Ask follow-up questions, compare alternatives, and iterate on your current design in conversation."
    )

    agent = load_conversational_agent()
    st.session_state.setdefault("chat_history", [])

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask anything about your structural design...")
    if not prompt:
        st.info("Try: `Design a 30ft beam with 1.2 kip/ft dead load and 1.8 kip/ft live load`, then ask `what if the span is 25ft?`")
        return

    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(prompt)
        st.write(response)

    st.session_state["chat_history"].extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    )


# --- Sidebar navigation ------------------------------------------------------
with st.sidebar:
    st.markdown("### Civil Agent")
    st.caption("Structural design intelligence")
    page = st.radio(
        "Navigate",
        ["Beam sizing", "RC beam (ACI 318)", "Column sizing", "Floor system", "Drawing review", "3D optimizer", "Generative design", "Plan intelligence", "Chat", "About"],
        label_visibility="collapsed",
    )

# --- About page --------------------------------------------------------------
if page == "About":
    st.title("About Civil Agent")
    st.markdown(
        f"""
        **Civil Agent** is a structural engineering assistant that sizes steel wide-flange (W-shape)
        **beams**, **columns**, and **floor systems** using **LRFD** load combinations
        (ASCE 7: Wu = 1.2D + 1.6L), **AISC 360**-style strength checks, **deflection** limits,
        **lateral torsional buckling (LTB)**, **P-M interaction**, **axial buckling**, and
        **local buckling** compactness.

        The tool combines:
        - A database of **{get_num_beams()} steel sections** with geometric properties  
          - **Physics engines** for beam, column, and floor system demand/capacity ratios  
          - **Reinforcement learning** agents trained to suggest efficient sections, verified by full check sets  
          - **Floor system optimizer** that finds the lightest steel bay for a given set of loads  
          - **ACI 318-19 concrete beam checks** for reinforced concrete flexure and shear with step-by-step traces  

        **Disclaimer:** This software is for education and preliminary design. A licensed Professional
        Engineer must review any design used for construction.

        ---
        *Color theme: navy & steel -- built with Streamlit.*
        """
    )
    st.stop()

# --- Drawing review page -----------------------------------------------------
if page == "Drawing review":
    drawing_review_page()
    st.stop()

# --- Reinforced concrete beam page -------------------------------------------
if page == "RC beam (ACI 318)":
    concrete_beam_page()
    st.stop()

# --- 3D optimizer page -------------------------------------------------------
if page == "3D optimizer":
    optimizer_3d_page()
    st.stop()

# --- Generative design page --------------------------------------------------
if page == "Generative design":
    generative_design_page()
    st.stop()

# --- Plan intelligence page --------------------------------------------------
if page == "Plan intelligence":
    plan_intelligence_page()
    st.stop()

if page == "Chat":
    conversational_chat_page()
    st.stop()

# --- Column sizing page -------------------------------------------------------
if page == "Column sizing":
    st.title("Column sizing")
    st.markdown(
        f'<p style="color:{STEEL};font-size:1.05rem;margin-bottom:1rem;">'
        "Describe your column in plain English, or use the structured form below."
        "</p>",
        unsafe_allow_html=True,
    )

    col_input_mode = st.radio(
        "Input method",
        ["Natural language", "Structured form"],
        horizontal=True,
        key="col_input_mode",
    )

    col_nl_text = ""
    col_h_v = 14.0
    col_pu_v = 300.0
    col_mu_v = 0.0
    col_K_v = 1.0

    if col_input_mode == "Natural language":
        col_nl_text = st.text_area(
            "Describe your column design problem",
            height=120,
            placeholder='e.g. 14ft tall column, 300 kips axial, 100 kip-ft moment',
            key="col_nl",
        )
        st.caption(
            "Default K = 1.0 (pinned-pinned). "
            'Specify "K=0.65" or "K=2.0" to override.'
        )
    else:
        cc1, cc2 = st.columns(2)
        with cc1:
            col_h_v = st.number_input(
                "Unbraced height (ft)", min_value=1.0, max_value=80.0,
                value=14.0, step=1.0, key="col_h")
        with cc2:
            col_pu_v = st.number_input(
                "Factored axial load Pu (kips)", min_value=0.0,
                value=300.0, step=10.0, key="col_pu")
        cc3, cc4 = st.columns(2)
        with cc3:
            col_mu_v = st.number_input(
                "Factored moment Mu (kip-ft)", min_value=0.0,
                value=0.0, step=10.0, key="col_mu")
        with cc4:
            col_K_v = st.selectbox(
                "Effective length factor K",
                options=[0.65, 0.80, 1.0, 1.2, 2.0],
                index=2,
                key="col_K",
            )

    col_design_clicked = st.button("Design column", type="primary", key="col_btn")

    if not col_design_clicked:
        st.info(
            "Enter loads and click **Design column**. "
            "First load trains the RL agent (~2 min) -- please wait."
        )
        st.stop()

    try:
        with st.spinner("Loading Civil Agent -- training RL policies on first run, please wait..."):
            agent = load_civil_agent()
    except Exception as e:
        st.error(f"Could not initialize the agent: {e}")
        st.stop()

    if col_input_mode == "Natural language":
        if not col_nl_text.strip():
            st.error("Please enter a description of your column problem.")
            st.stop()
        params = parse(col_nl_text.strip(), use_claude=False)
        if params.get('type') != 'column' or params.get('height_ft') is None:
            st.error(
                "Could not parse column parameters. "
                "Try: *14ft tall column, 300 kips axial, 100 kip-ft moment*."
            )
            st.stop()
        height_ft  = float(params['height_ft'])
        Pu_kips    = float(params.get('axial_load', 200))
        Mu_kipft   = float(params.get('moment', 0))
        K_factor   = float(params.get('K_factor', 1.0))
    else:
        height_ft  = col_h_v
        Pu_kips    = col_pu_v
        Mu_kipft   = col_mu_v
        K_factor   = col_K_v

    try:
        with st.spinner("Running column checks..."):
            result = agent.find_optimal_column(
                height_ft=height_ft,
                Pu_kips=Pu_kips,
                Mu_kipft=Mu_kipft,
                K=K_factor,
            )
    except Exception as e:
        st.error(f"Column design failed: {e}")
        st.stop()

    if not result.get("details"):
        st.error(
            "No section in the database passes all column checks for these inputs. "
            "Try reducing loads or height."
        )
        st.stop()

    d = result["details"]
    col_name = result["column_name"]
    passes = result["passes"]
    best_action = find_column_action_index(col_name)
    col_row = COLUMN_SECTIONS.iloc[best_action]
    col_depth = float(col_row["d"])
    total_col_weight = float(result["weight"]) * float(height_ft)

    st.markdown("---")

    st.subheader("Applied loads")
    st.markdown(
        f"**Pu** = **{Pu_kips:g}** kips  \n"
        f"**Mu** = **{Mu_kipft:g}** kip-ft  \n"
        f"**K** = **{K_factor:g}**  |  "
        f"KL/r = **{d['KL_r']:.1f}**  |  "
        f"Buckling: **{d['buckling_type']}**"
    )

    badge = "PASS" if passes else "FAIL"
    badge_color = "#2a9d8f" if passes else "#e63946"
    st.markdown(
        f"""
        <div style="background:{NAVY};border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;
             box-shadow:0 4px 14px rgba(13,27,42,0.25);">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
                <div>
                    <div style="color:#adb5bd;font-size:0.9rem;text-transform:uppercase;letter-spacing:0.06em;">
                        Optimal section</div>
                    <div style="color:#fff;font-size:2.1rem;font-weight:700;">{col_name}</div>
                    <div style="color:#e0e1dd;font-size:0.95rem;margin-top:0.35rem;">
                        Weight: {result["weight"]:.1f} lb/ft<br>
                        Depth: {col_depth:.1f} in<br>
                        Total column weight: {total_col_weight:,.0f} lb</div>
                </div>
                <div style="background:{badge_color};color:white;padding:0.5rem 1.25rem;border-radius:8px;
                     font-weight:800;font-size:1.1rem;letter-spacing:0.05em;">{badge}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not str(col_name).startswith("W14"):
        w14_pick = find_lightest_passing_w14(
            height_ft, Pu_kips, Mu_kipft, K_factor)
        if w14_pick:
            st.caption(
                "W14 sections are standard for multi-story column lines. "
                f"Nearest W14: **{w14_pick['name']}** "
                f"(max utilization {w14_pick['max_ratio']:.2f})."
            )
        else:
            st.caption(
                "W14 sections are standard for multi-story column lines. "
                "No W14 in the catalog passes these checks at this size — "
                "review loads, K-factor, or use a built-up section."
            )

    st.subheader("Capacity summary")
    st.markdown(
        f"**phi_Pn** = {d['phi_Pn']:.1f} kips  |  "
        f"**phi_Mn** = {d['phi_Mn']:.1f} kip-ft  |  "
        f"**Fcr** = {d['Fcr']:.1f} ksi  |  "
        f"Interaction eq: **{d['pm_equation']}**"
    )

    rows = []
    for key, label in [("axial_ratio", "Axial capacity"),
                       ("pm_ratio", "P-M interaction"),
                       ("flange_ratio", "Local buckl. (flange)"),
                       ("web_ratio", "Local buckl. (web)")]:
        r = float(d[key])
        icon = "OK" if r <= 1.0 else "FAIL"
        rows.append({"Check": label, "Ratio": f"{r:.2f}", "Status": icon})

    df_col = pd.DataFrame(rows)
    st.dataframe(df_col, use_container_width=True, hide_index=True)

    fig = column_utilization_chart(d)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Explanation")
    st.markdown(result["explanation"].replace("\n", "\n\n"))

    alt_lines = col_neighbor_alternatives(
        height_ft, Pu_kips, Mu_kipft, K_factor, best_action)
    if alt_lines:
        st.subheader("Alternatives")
        for title, text in alt_lines:
            st.markdown(f"**{title}:** {text}")

    st.caption("Ratios <= 1.0 satisfy the corresponding limit state.")
    st.stop()

# --- Floor system page --------------------------------------------------------
if page == "Floor system":
    st.title("Floor System Design")
    st.markdown(
        f'<p style="color:{STEEL};font-size:1.05rem;margin-bottom:1rem;">'
        "Design beams, girders, and columns for a rectangular steel bay. "
        "Sequential: beams first, then girders from beam reactions, then columns."
        "</p>",
        unsafe_allow_html=True,
    )

    _fc1, _fc2 = st.columns(2)
    with _fc1:
        fs_bay_len = st.number_input(
            "Bay length (ft) — girder span",
            min_value=10.0, max_value=100.0, value=40.0, step=5.0,
            key="fs_bl")
        fs_dl = st.number_input(
            "Dead load (psf)", min_value=5.0, max_value=300.0,
            value=50.0, step=5.0, key="fs_dl")
        fs_nf = st.number_input(
            "Floors above column",
            min_value=1, max_value=50, value=1, step=1, key="fs_nf")
    with _fc2:
        fs_bay_w = st.number_input(
            "Bay width (ft) — beam span",
            min_value=10.0, max_value=80.0, value=30.0, step=5.0,
            key="fs_bw")
        fs_ll = st.number_input(
            "Live load (psf)", min_value=10.0, max_value=500.0,
            value=80.0, step=5.0, key="fs_ll")
        fs_fh = st.number_input(
            "Floor-to-floor height (ft)",
            min_value=8.0, max_value=30.0, value=14.0, step=1.0,
            key="fs_fh")

    _fc3, _fc4 = st.columns(2)
    with _fc3:
        fs_sp = st.number_input(
            "Beam spacing (ft)", min_value=3.0, max_value=30.0,
            value=10.0, step=1.0, key="fs_sp")
    with _fc4:
        fs_defl = st.selectbox(
            "Deflection limit", options=[240, 360, 480], index=1,
            format_func=lambda x: f"L/{x}", key="fs_defl")

    fs_opt = st.checkbox(
        "Optimize beam spacing (try 5, 8, 10, 12, 15 ft)", key="fs_opt")

    fs_btn = st.button("Design Floor System", type="primary", key="fs_btn")

    if not fs_btn:
        st.info(
            "Set bay dimensions and loads, then click **Design Floor System**. "
            "No RL training needed — results are instant."
        )
        st.stop()

    with st.spinner("Designing floor system..."):
        from floor_system import FloorSystem  # noqa: E402

        if fs_opt:
            from floor_optimizer import optimize_floor  # noqa: E402
            _fs_best, _fs_best_sp, _fs_all = optimize_floor(
                fs_bay_len, fs_bay_w, fs_dl, fs_ll,
                num_floors=fs_nf, floor_height_ft=fs_fh,
                defl_limit=fs_defl)
            if _fs_best is None:
                st.error(
                    "No beam spacing produces a passing design. "
                    "Try reducing loads or bay dimensions.")
                st.stop()
            _fs_res = _fs_best['results']
            _fs_used_sp = _fs_best_sp
        else:
            _fs_obj = FloorSystem(
                fs_bay_len, fs_bay_w, fs_dl, fs_ll,
                beam_spacing_ft=fs_sp, num_floors=fs_nf,
                floor_height_ft=fs_fh, defl_limit=fs_defl)
            _fs_res = _fs_obj.design_all()
            _fs_used_sp = fs_sp
            _fs_all = None
            _fs_best_sp = None

    _fb = _fs_res['beams']
    _fg = _fs_res['girder']
    _fco = _fs_res['column']

    st.markdown("---")

    # ---- Three result cards ----
    def _fs_card(title, name, passes, lines):
        badge = "PASS" if passes else "FAIL"
        bc = "#2a9d8f" if passes else "#e63946"
        body = "<br>".join(lines)
        return (
            f'<div style="background:{NAVY};border-radius:12px;padding:1.2rem 1.4rem;'
            f'min-height:220px;box-shadow:0 4px 14px rgba(13,27,42,0.25);">'
            f'<div style="color:#adb5bd;font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.06em;">{title}</div>'
            f'<div style="color:#fff;font-size:1.7rem;font-weight:700;'
            f'margin:0.3rem 0;">{name}</div>'
            f'<div style="color:#e0e1dd;font-size:0.88rem;line-height:1.55;">'
            f'{body}</div>'
            f'<div style="background:{bc};color:white;display:inline-block;'
            f'padding:0.2rem 0.8rem;border-radius:6px;font-weight:700;'
            f'font-size:0.85rem;margin-top:0.5rem;">{badge}</div></div>'
        )

    _cc1, _cc2, _cc3 = st.columns(3)
    with _cc1:
        st.markdown(_fs_card("Beams", _fb['name'], _fb['passes'], [
            f"{_fb['span']:g}ft span",
            f"{_fb['num_beams']} @ {_fs_used_sp:g}ft o.c.",
            f"{_fb['weight']:.1f} lb/ft",
            f"Reaction: {_fb['reaction']:.1f} kips",
        ]), unsafe_allow_html=True)
    with _cc2:
        st.markdown(_fs_card("Girder", _fg['name'], _fg['passes'], [
            f"{_fg['span']:g}ft span",
            f"{_fg['n_point_loads']} pt loads @ {_fg['point_load_kips']:.1f}k",
            f"{_fg['weight']:.1f} lb/ft",
            f"Reaction: {_fg['reaction']:.1f} kips",
        ]), unsafe_allow_html=True)
    with _cc3:
        st.markdown(_fs_card("Column", _fco['name'], _fco['passes'], [
            f"{_fco['height']:g}ft tall, K={_fco['K']}",
            f"Pu = {_fco['Pu']:.1f} kips",
            f"Mu = {_fco['Mu']:.1f} kip-ft",
            f"{_fco['weight']:.1f} lb/ft",
        ]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Total weight metrics ----
    _bwt = _fb['weight'] * _fb['span'] * _fb['num_beams']
    _gwt = _fg['weight'] * _fg['span']
    _cwt = _fco['weight'] * _fco['height']
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Total Steel", f"{_fs_res['total_weight']:,.0f} lbs")
    _m2.metric("Beams", f"{_bwt:,.0f} lbs")
    _m3.metric("Girder", f"{_gwt:,.0f} lbs")
    _m4.metric("Column", f"{_cwt:,.0f} lbs")

    _savings = _fs_res.get("composite_savings")
    if _savings:
        st.subheader("Composite Savings")
        _sm1, _sm2, _sm3, _sm4 = st.columns(4)
        _sm1.metric(
            "Beam savings",
            f"{_savings.get('beam_savings_lbs', 0):,.0f} lbs",
            delta=f"{_savings.get('beam_noncomp_lbs', 0):,.0f} -> "
                  f"{_savings.get('beam_composite_lbs', 0):,.0f}",
        )
        _sm2.metric(
            "Girder savings",
            f"{_savings.get('girder_savings_lbs', 0):,.0f} lbs",
            delta=f"{_savings.get('girder_noncomp_lbs', 0):,.0f} -> "
                  f"{_savings.get('girder_composite_lbs', 0):,.0f}",
        )
        _sm3.metric(
            "Total savings",
            f"{_savings.get('total_savings_lbs', 0):,.0f} lbs",
        )
        _sm4.metric(
            "Cost savings",
            f"${_savings.get('cost_savings', 0):,.0f}",
        )
        st.dataframe(
            pd.DataFrame([
                {
                    "Member": "Beams",
                    "Composite steel (lbs)": _savings.get("beam_composite_lbs", 0),
                    "Non-composite steel (lbs)": _savings.get("beam_noncomp_lbs", 0),
                    "Savings (lbs)": _savings.get("beam_savings_lbs", 0),
                },
                {
                    "Member": "Girder",
                    "Composite steel (lbs)": _savings.get("girder_composite_lbs", 0),
                    "Non-composite steel (lbs)": _savings.get("girder_noncomp_lbs", 0),
                    "Savings (lbs)": _savings.get("girder_savings_lbs", 0),
                },
            ]),
            use_container_width=True,
            hide_index=True,
        )

    _slab = _fs_res.get("slab") or {}
    _footing = _fs_res.get("footing") or {}
    _base_plate = _fs_res.get("base_plate") or {}
    if _slab or _footing or _base_plate:
        st.subheader("Slab and Foundation")
        sf1, sf2, sf3 = st.columns(3)
        with sf1:
            st.metric("Deck", _slab.get("deck_type", "N/A"))
            st.metric("Slab thickness", f"{_slab.get('total_thickness_in', 0):.1f} in")
            st.metric("Fire rating", f"{_slab.get('fire_rating_hrs', 0)} hr")
            st.caption(_slab.get("WWF_designation", ""))
        with sf2:
            st.metric("Footing", f"{_footing.get('B_ft', 0):.1f}ft x {_footing.get('L_ft', 0):.1f}ft")
            st.metric("Footing thickness", f"{_footing.get('h_in', 0):.0f} in")
            st.metric("Soil pressure", f"{_footing.get('qu_ksf', 0):.2f} ksf")
            st.caption(
                f"{_footing.get('bar_size', '#5')} @ {_footing.get('bar_spacing_in', 0):.1f} in"
            )
        with sf3:
            st.metric("Base plate", f"{_base_plate.get('N_in', 0):.0f} x {_base_plate.get('B_in', 0):.0f} in")
            st.metric("Plate thickness", f"{_base_plate.get('t_in', 0):.2f} in")
            st.metric("Estimated foundation cost", f"${(_footing.get('cost_estimate', 0) + _base_plate.get('cost_estimate', 0)):,.0f}")
            st.caption(_footing["notes"][0] if _footing.get("notes") else "Representative spread footing and base plate.")

    # ---- Load path diagram ----
    st.subheader("Load Path")
    _total_psf = fs_dl + fs_ll
    _nf_label = f"{fs_nf} floor{'s' if fs_nf > 1 else ''}"
    st.markdown(
        f"""
        <div style="background:#f0f2f6;border-radius:10px;padding:1.2rem 1.8rem;
             text-align:center;font-size:0.95rem;line-height:2.2;">
            <b>Floor load:</b> {_total_psf:g} psf
            ({fs_dl:g} DL + {fs_ll:g} LL)<br>
            <span style="font-size:1.4rem;">&#x2193;</span><br>
            <b>Beams:</b> {_fb['num_beams']} &times; {_fb['name']}
            @ {_fs_used_sp:g}ft o.c.<br>
            <span style="font-size:1.4rem;">&#x2193;</span>
            <span style="color:#555;font-size:0.85rem;">
            &nbsp;{_fb['reaction']:.1f}k per beam end</span><br>
            <b>Girder:</b> {_fg['name']}
            ({_fg['n_point_loads']} &times; {_fg['point_load_kips']:.1f}k
            pt loads)<br>
            <span style="font-size:1.4rem;">&#x2193;</span>
            <span style="color:#555;font-size:0.85rem;">
            &nbsp;{_fg['reaction']:.1f}k per girder end</span><br>
            <b>Column:</b> {_fco['name']}
            ({_fco['Pu']:.1f}k axial, {_nf_label})
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Optimization table ----
    if _fs_all is not None:
        st.subheader("Spacing Comparison")
        _opt_rows = []
        for _r in _fs_all:
            _status = "PASS" if _r['passes'] else "FAIL"
            if _r['passes'] and _r['spacing'] == _fs_best_sp:
                _status = "PASS (optimal)"
            _opt_rows.append({
                'Spacing (ft)': f"{_r['spacing']:g}",
                'Beam': _r['beam'],
                'Girder': _r['girder'],
                'Column': _r['column'],
                'Total (lbs)': f"{_r['total_weight']:,.0f}",
                'Status': _status,
            })
        _opt_df = pd.DataFrame(_opt_rows)
        st.dataframe(_opt_df, use_container_width=True, hide_index=True)
        st.caption(f"Optimal spacing: **{_fs_best_sp:g} ft** "
                   f"(lightest passing total: "
                   f"{_fs_res['total_weight']:,.0f} lbs)")

    # ---- Detailed checks in expanders ----
    if _fb.get('details'):
        with st.expander("Beam check details"):
            _br = []
            for _key, _lbl in [('moment_ratio', 'Moment'),
                                ('shear_ratio', 'Shear'),
                                ('defl_ratio', 'Deflection'),
                                ('ltb_ratio', 'LTB'),
                                ('flange_ratio', 'Flange buckling'),
                                ('web_ratio', 'Web buckling')]:
                _v = float(_fb['details'].get(_key, 0))
                _s = 'OK' if _v <= 1.0 else 'FAIL'
                _e = (f" ({_fb['details'].get('ltb_zone', '')})"
                      if _key == 'ltb_ratio' else "")
                _br.append({'Check': _lbl, 'Ratio': f'{_v:.3f}',
                            'Status': f'{_s}{_e}'})
            st.dataframe(pd.DataFrame(_br),
                         use_container_width=True, hide_index=True)
            _bfig = utilization_chart(_fb['details'])
            st.pyplot(_bfig, use_container_width=True)
            plt.close(_bfig)

    if _fg.get('details'):
        with st.expander("Girder check details"):
            _gr = []
            for _key, _lbl in [('moment_ratio', 'Moment'),
                                ('shear_ratio', 'Shear'),
                                ('defl_ratio', 'Deflection'),
                                ('ltb_ratio', 'LTB'),
                                ('flange_ratio', 'Flange buckling'),
                                ('web_ratio', 'Web buckling')]:
                _v = float(_fg['details'].get(_key, 0))
                _s = 'OK' if _v <= 1.0 else 'FAIL'
                _e = (f" ({_fg['details'].get('ltb_zone', '')})"
                      if _key == 'ltb_ratio' else "")
                _gr.append({'Check': _lbl, 'Ratio': f'{_v:.3f}',
                            'Status': f'{_s}{_e}'})
            st.dataframe(pd.DataFrame(_gr),
                         use_container_width=True, hide_index=True)
            _gfig = utilization_chart(_fg['details'])
            st.pyplot(_gfig, use_container_width=True)
            plt.close(_gfig)

    if _fco.get('details'):
        with st.expander("Column check details"):
            _cr = []
            for _key, _lbl in [('axial_ratio', 'Axial capacity'),
                                ('pm_ratio', 'P-M interaction'),
                                ('flange_ratio', 'Flange buckling'),
                                ('web_ratio', 'Web buckling')]:
                _v = float(_fco['details'].get(_key, 0))
                _s = 'OK' if _v <= 1.0 else 'FAIL'
                _cr.append({'Check': _lbl, 'Ratio': f'{_v:.3f}',
                            'Status': _s})
            st.dataframe(pd.DataFrame(_cr),
                         use_container_width=True, hide_index=True)
            _cfig = column_utilization_chart(_fco['details'])
            st.pyplot(_cfig, use_container_width=True)
            plt.close(_cfig)

    st.caption("Ratios <= 1.0 satisfy the corresponding limit state.")
    st.stop()

# --- Main: Beam sizing -------------------------------------------------------
st.title("Beam sizing")
st.markdown(
    f'<p style="color:{STEEL};font-size:1.05rem;margin-bottom:1rem;">'
    "Describe your problem in plain English, or use the structured form below."
    "</p>",
    unsafe_allow_html=True,
)

input_mode = st.radio(
    "Input method",
    ["Natural language", "Structured form"],
    horizontal=True,
)

nl_text = ""
span_v = dead_v = live_v = 20.0
lb_v = 0.0
point_v = 0.0
defl_choice = 360

if input_mode == "Natural language":
    nl_text = st.text_area(
        "Describe your beam sizing problem",
        height=120,
        placeholder=(
            "e.g. 30ft span, 1.2 kip/ft dead load, 1.8 kip/ft live load, "
            "unbraced length 15ft"
        ),
    )
    st.caption(
        'Assumed unbraced length = full span (conservative). '
        'Specify "unbraced length Xft" or "fully braced" to override.'
    )
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        span_v = st.number_input("Span (ft)", min_value=1.0, max_value=200.0, value=30.0, step=1.0)
    with c2:
        dead_v = st.number_input("Dead load (kip/ft)", min_value=0.0, value=1.2, step=0.1, format="%.2f")
    with c3:
        live_v = st.number_input("Live load (kip/ft)", min_value=0.0, value=1.8, step=0.1, format="%.2f")
    c4, c5, c6 = st.columns(3)
    with c4:
        lb_v = st.number_input(
            "Unbraced length (ft), optional",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="0 = use full span as unbraced length",
        )
        st.caption(
            'Assumed unbraced length = full span (conservative). '
            'Specify "unbraced length Xft" or "fully braced" to override.'
        )
    with c5:
        point_v = st.number_input("Point load at midspan (kips)", min_value=0.0, value=0.0, step=0.5)
    with c6:
        defl_choice = st.selectbox(
            "Deflection limit",
            options=[240, 360, 480],
            index=1,
            format_func=lambda x: f"L/{x}",
        )

st.markdown("---")
_comp_enabled = st.checkbox(
    "Composite beam (steel + concrete slab on metal deck)",
    key="beam_composite_chk",
)
if _comp_enabled:
    _cp1, _cp2, _cp3, _cp4 = st.columns(4)
    with _cp1:
        _comp_tc = st.number_input(
            "Slab thickness above deck (in)",
            min_value=2.0, max_value=6.0, value=3.5, step=0.5,
            key="comp_tc")
    with _cp2:
        _comp_fc = st.number_input(
            "Concrete strength fc (ksi)",
            min_value=2.5, max_value=8.0, value=4.0, step=0.5,
            key="comp_fc")
    with _cp3:
        _comp_ratio = st.slider(
            "Target composite ratio (%)",
            min_value=25, max_value=100, value=50, step=5,
            key="comp_ratio") / 100.0
    with _cp4:
        _comp_sp = st.number_input(
            "Beam spacing (ft)",
            min_value=3.0, max_value=20.0, value=10.0, step=1.0,
            key="comp_sp")
    _comp_shored = st.checkbox(
        "Shored construction",
        value=False,
        help=(
            "Temporary shores support beam weight during construction. "
            "Allows lighter sections but adds construction cost."
        ),
        key="comp_shored",
    )
    st.caption(
        "Deck height assumed 3 in. Unshored construction uses beam spacing as deck-braced Lb. "
        "Deflection controlled by live load only using composite Ieff."
    )
else:
    _comp_tc = 3.5
    _comp_fc = 4.0
    _comp_ratio = 0.5
    _comp_sp = 10.0
    _comp_shored = False

design_clicked = st.button("Design beam", type="primary")

if not design_clicked:
    st.info("Enter your loads and click **Design beam**. First load trains the RL agent (~1 minute) — please wait.")
    st.stop()

# Load agent (cached after first run)
try:
    with st.spinner("Loading Civil Agent — training RL policy on first run only, please wait…"):
        agent = load_civil_agent()
except Exception as e:
    st.error(f"Could not initialize the agent: {e}")
    st.stop()

# Resolve parameters
if input_mode == "Natural language":
    if not nl_text.strip():
        st.error("Please enter a description of your beam problem.")
        st.stop()
    params = parse(nl_text.strip(), use_claude=False)
    if params.get("span_ft") is None:
        st.error(
            "Could not find a span in your text. "
            "Try: *30ft span, 1.2 kip/ft dead, 1.8 kip/ft live*."
        )
        st.stop()
    span_ft = float(params["span_ft"])
    dead_load = float(params["dead_load"])
    live_load = float(params["live_load"])
    point_load = float(params.get("point_load") or 0)
    defl_limit = int(params.get("defl_limit") or 360)
    lb_raw = params.get("Lb_ft")
    if lb_raw is None:
        lb_use = None
    else:
        lb_use = float(lb_raw)
else:
    span_ft = span_v
    dead_load = dead_v
    live_load = live_v
    point_load = point_v
    defl_limit = defl_choice
    if lb_v <= 0:
        lb_use = None
    else:
        lb_use = lb_v

Wu = apply_lrfd(dead_load, live_load)
Mu, Vu, _ = calculate_demands(span_ft, dead_load, live_load, point_load)
lb_display = span_ft if lb_use is None else lb_use

try:
    with st.spinner("Running structural checks…"):
        result = agent.find_optimal_beam(
            span_ft=span_ft,
            dead_load=dead_load,
            live_load=live_load,
            Lb_ft=lb_use,
            point_load=point_load,
            defl_limit=defl_limit,
            composite=_comp_enabled,
            beam_spacing_ft=_comp_sp,
            slab_thickness_in=_comp_tc,
            fc_ksi=_comp_fc,
            composite_ratio=_comp_ratio,
            shored=_comp_shored,
        )
except Exception as e:
    st.error(f"Design check failed: {e}")
    st.stop()

_is_composite_result = result.get("composite", False)

if not result.get("details"):
    st.error(
        "No beam in the database satisfied all limit states for these inputs. "
        "Try a shorter span, lower loads, or relax the deflection limit."
    )
    st.stop()

d = result["details"]
beam_name = result["beam_name"]
passes = result["passes"]
best_action = find_best_action_index(beam_name)
beam_row = BEAMS_DF.iloc[best_action]
beam_depth = float(beam_row["d"])
total_beam_weight = float(result["weight"]) * float(span_ft)

alt_lines = [] if _is_composite_result else neighbor_alternatives(
    span_ft, dead_load, live_load, lb_use, point_load, defl_limit, best_action
)

# --- Result card -------------------------------------------------------------
st.markdown("---")

st.subheader("Applied loads")
if _is_composite_result:
    st.markdown(
        f"**Wu** = 1.2({dead_load:g}) + 1.6({live_load:g}) = **{Wu:.2f}** kip/ft  \n"
        f"**Mu** = {Wu:.2f} × {span_ft:g}² / 8 = **{Mu:.2f}** kip-ft  \n"
        f"**Composite ratio:** {int(_comp_ratio*100)}%  |  "
        f"**fc** = {_comp_fc} ksi  |  "
        f"**slab tc** = {_comp_tc} in  |  "
        f"**beam spacing** = {_comp_sp:g} ft"
    )
else:
    st.markdown(
        f"**Wu** = 1.2({dead_load:g}) + 1.6({live_load:g}) = **{Wu:.2f}** kip/ft  \n"
        f"**Mu** = {Wu:.2f} × {span_ft:g}² / 8"
        + (f" + 1.2({point_load:g}) × {span_ft:g} / 4" if point_load else "")
        + f" = **{Mu:.2f}** kip-ft  \n"
        f"**Vu** = {Wu:.2f} × {span_ft:g} / 2"
        + (f" + 1.2({point_load:g}) / 2" if point_load else "")
        + f" = **{Vu:.2f}** kips  \n"
        + f"**Unbraced length used for LTB:** **{lb_display:g} ft**"
    )

badge = "PASS" if passes else "FAIL"
badge_color = "#2a9d8f" if passes else "#e63946"
st.markdown(
    f"""
    <div style="background:{NAVY};border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;
         box-shadow:0 4px 14px rgba(13,27,42,0.25);">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
            <div>
                <div style="color:#adb5bd;font-size:0.9rem;text-transform:uppercase;letter-spacing:0.06em;">
                    Optimal section</div>
                <div style="color:#fff;font-size:2.1rem;font-weight:700;">{beam_name}</div>
                <div style="color:#e0e1dd;font-size:0.95rem;margin-top:0.35rem;">
                    Weight: {result["weight"]:.1f} lb/ft<br>
                    Depth: {beam_depth:.1f} in<br>
                    Total beam weight: {total_beam_weight:,.0f} lb</div>
            </div>
            <div style="background:{badge_color};color:white;padding:0.5rem 1.25rem;border-radius:8px;
                 font-weight:800;font-size:1.1rem;letter-spacing:0.05em;">{badge}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Composite design info panel -------------------------------------------
if _is_composite_result:
    _savings = result.get("steel_savings", 0.0)
    _savings_lbs = _savings * span_ft
    _cost_save = _savings_lbs * 1.50
    _nc_wt = result.get("nc_weight", result["weight"])
    _Ieff = result.get("Ieff", 0)
    _Is = result.get("Is", 1)
    _Itr = result.get("Itr", 0)
    _stiff_gain = round((_Ieff / _Is - 1) * 100) if _Is > 0 else 0

    st.subheader("Composite Design")
    _cm1, _cm2, _cm3 = st.columns(3)
    _cm1.metric("Ieff (composite)", f"{_Ieff:.0f} in\u2074",
                delta=f"+{_stiff_gain:.0f}% vs bare steel")
    _cm2.metric("Studs per side", result.get("studs_per_side", 0),
                delta=f"{result.get('stud_count', 0)} total")
    _cm3.metric("Qn per stud", f"{result.get('Qn_per_stud', 0):.1f} kips")

    _ccd1, _ccd2 = st.columns(2)
    with _ccd1:
        st.markdown(
            f"**beff** = {result.get('beff_in', 0):.1f} in  |  "
            f"**n** = {result['details'].get('n_modular', 8):.0f}  |  "
            f"**Cf** = {result.get('Cf_kips', 0):.1f} kips  \n"
            f"**Is** = {_Is:.0f} in\u2074  |  "
            f"**Ieff** = {_Ieff:.0f} in\u2074  |  "
            f"**Itr** = {_Itr:.0f} in\u2074"
        )
        _comp_check_rows = [
            {"Check": "Construction (steel alone)", "Ratio":
             f"{result['details'].get('construction_ratio', 0):.3f}",
             "Status": "OK" if result['details'].get('construction_ratio', 0) <= 1.0 else "FAIL"},
            {"Check": "Composite moment", "Ratio":
             f"{result['details'].get('moment_ratio', 0):.3f}",
             "Status": "OK" if result['details'].get('moment_ratio', 0) <= 1.0 else "FAIL"},
            {"Check": "Live-load deflection (Ieff)", "Ratio":
             f"{result['details'].get('deflection_ratio', 0):.3f}",
             "Status": "OK" if result['details'].get('deflection_ratio', 0) <= 1.0 else "FAIL"},
        ]
        st.dataframe(pd.DataFrame(_comp_check_rows),
                     use_container_width=True, hide_index=True)

    with _ccd2:
        _save_color = "#2a9d8f" if _savings > 0 else "#e63946"
        _save_label = "saves" if _savings > 0 else "costs extra"
        _save_abs = abs(_savings)
        _save_lbs_abs = abs(_savings_lbs)
        _cost_abs = abs(_cost_save)
        st.markdown(
            f"""
            <div style="background:{_save_color};border-radius:10px;
            padding:1rem 1.4rem;color:white;">
            <div style="font-size:0.85rem;text-transform:uppercase;
            letter-spacing:0.05em;opacity:0.85;">Steel vs non-composite</div>
            <div style="font-size:1.6rem;font-weight:700;margin:0.3rem 0;">
            {_save_abs:.1f} lb/ft {_save_label}</div>
            <div style="font-size:0.9rem;opacity:0.9;">
            Non-composite: {_nc_wt:.1f} lb/ft<br>
            Composite: {result["weight"]:.1f} lb/ft<br>
            Total: {_save_lbs_abs:,.0f} lbs &nbsp;~&nbsp;
            ${_cost_abs:,.0f} at $1.50/lb</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Utilization table
rows = []
for key, label in CHECK_LABELS:
    r = float(d[key])
    st_icon = "✓" if r <= 1.0 else "✗"
    extra = ""
    if key == "ltb_ratio":
        extra = f" ({d.get('ltb_zone', '')})"
    rows.append({"Check": label, "Ratio": f"{r:.2f}", "Status": f"{st_icon}{extra}"})

lb_u = _local_buckling_util(d)
rows.append(
    {
        "Check": "Local buckling",
        "Ratio": f"{lb_u:.2f}",
        "Status": "✓" if lb_u <= 1.0 else "✗",
    }
)

df = pd.DataFrame(rows)
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

fig = utilization_chart(d)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

conn = result.get("connection")
if conn is not None:
    st.divider()
    st.subheader("Connection design")
    st.caption("Shear tab - beam web to column flange (automatic)")

    bl = conn.get("bolt_label", '3/4"')
    st.markdown(
        f"### {conn['num_bolts']} - {bl} A325-N bolts"
    )
    st.markdown(
        f"**Tab:** {conn['tab_thickness']}\" × "
        f"{conn['tab_length']}\" A36 shear tab"
    )
    st.markdown("**Weld:** 3/16\" fillet both sides to column flange")

    conn_checks = {
        "Bolt shear": conn["checks"]["bolt_shear"],
        "Bolt bearing": conn["checks"]["bolt_bearing"],
        "Tab shear": conn["checks"]["tab_shear"],
        "Weld capacity": conn["checks"]["weld"],
    }
    conn_rows = [
        {"Check": k, "Ratio": f"{v:.2f}", "Status": "OK" if v <= 1.0 else "FAIL"}
        for k, v in conn_checks.items()
    ]
    st.dataframe(
        pd.DataFrame(conn_rows),
        use_container_width=True,
        hide_index=True,
    )

    cfig = connection_utilization_chart(conn)
    st.pyplot(cfig, use_container_width=True)
    plt.close(cfig)

    if conn.get("passes", False):
        st.info(conn["explanation"])
    else:
        st.warning(conn["explanation"])

st.subheader("ML Section Recommendations")
ml_recs = predict_section(
    span_ft=span_ft,
    dead_load=dead_load,
    live_load=live_load,
    beam_spacing=_comp_sp if _comp_enabled else 10.0,
    top_k=3,
)
if ml_recs:
    st.dataframe(
        pd.DataFrame(
            [{"Section": name, "Confidence": confidence} for name, confidence in ml_recs]
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("No trained section recommender is available yet, so Civil Agent is using the deterministic scan only.")

st.subheader("Moment connection design")
_mc1, _mc2 = st.columns(2)
with _mc1:
    moment_conn_type = st.selectbox("Connection type", ["BFP", "EEP", "WUF-W"], key="beam_moment_conn_type")
with _mc2:
    moment_column_name = st.selectbox(
        "Column section",
        COLUMN_SECTIONS["name"].tolist(),
        index=min(10, len(COLUMN_SECTIONS) - 1),
        key="beam_moment_column",
    )
moment_column = COLUMN_SECTIONS[COLUMN_SECTIONS["name"] == moment_column_name].iloc[0].to_dict()
moment_connection = design_moment_connection(
    beam=beam_row.to_dict(),
    column=moment_column,
    Mu_kip_ft=Mu,
    Vu_kips=float(result.get("Vu", Vu)),
    conn_type=moment_conn_type,
)
result["moment_connection"] = moment_connection
st.dataframe(
    pd.DataFrame(
        [
            {"Metric": "Type", "Value": moment_connection["type"]},
            {"Metric": "Flange force", "Value": f"{moment_connection['flange_force_kips']:.1f} kips"},
            {"Metric": "Plate", "Value": f"{moment_connection['plate_width_in']:.1f} in x {moment_connection['plate_thickness_in']:.3f} in" if moment_connection["plate_thickness_in"] else "N/A"},
            {"Metric": "Bolts per flange", "Value": moment_connection["num_bolts_flange"]},
            {"Metric": "Weld size", "Value": f"{moment_connection['weld_size_in']:.3f} in"},
            {"Metric": "Cost estimate", "Value": f"${moment_connection['cost_estimate']:,.0f}"},
            {"Metric": "Status", "Value": "PASS" if moment_connection["passes"] else "FAIL"},
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
if not moment_connection.get("panel_zone_ok", True):
    st.warning("Panel zone check does not pass for the selected column; use a heavier column or a reinforced panel zone.")

st.subheader("Explanation")
st.markdown(result["explanation"].replace("\n", "\n\n"))

if alt_lines:
    st.subheader("Alternatives")
    for title, text in alt_lines:
        st.markdown(f"**{title}:** {text}")

try:
    pdf_bytes = generate_beam_report_pdf(
        result,
        span_ft=span_ft,
        dead_load=dead_load,
        live_load=live_load,
        point_load=point_load,
        defl_limit=defl_limit,
        lb_display=lb_display,
        Wu=Wu,
        Mu=Mu,
        Vu=float(result.get("Vu", Vu)),
    )
    st.download_button(
        label="Download PDF report",
        data=pdf_bytes,
        file_name="civil_agent_beam_report.pdf",
        mime="application/pdf",
        key="dl_beam_pdf",
    )
except Exception as pdf_err:
    st.caption(f"PDF export unavailable: {pdf_err}")

st.caption("Ratios ≤ 1.0 satisfy the corresponding limit state.")
