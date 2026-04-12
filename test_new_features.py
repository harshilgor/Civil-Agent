from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from beams_data import BEAMS_DF, COLUMN_SECTIONS
from civil_agent import ConversationalCivilAgent
from connection_design import design_moment_connection
from foundation_design import design_base_plate, design_spread_footing
from process_drawing_batch import process_drawing
from section_recommender import predict_section
from slab_design import check_deck_span, design_composite_slab


def test_slab_design_returns_complete_pass_result():
    result = design_composite_slab(
        beam_spacing_ft=10.0,
        total_slab_thickness_in=6.5,
        dead_load_psf=30.0,
        live_load_psf=80.0,
        deck_type="3B",
        fc_ksi=4.0,
    )
    assert result["deck_type"] == "3B"
    assert "WWF_designation" in result
    assert "passes" in result
    assert result["deck_span_check"]["passes"] is True


def test_footing_and_base_plate_return_structured_outputs():
    footing = design_spread_footing(Pu_kips=250.0, Mu_kip_ft=25.0)
    base_plate = design_base_plate(
        Pu_kips=250.0,
        Mu_kip_ft=25.0,
        column={"bf": 14.0, "d": 14.0, "tf": 0.75, "tw": 0.5},
    )
    assert footing["B_ft"] > 0
    assert footing["passes"] in {True, False}
    assert base_plate["t_in"] >= 0.75
    assert base_plate["passes"] is True


def test_moment_connection_design_returns_expected_schema():
    beam = BEAMS_DF.iloc[10].to_dict()
    column = COLUMN_SECTIONS.iloc[0].to_dict()
    result = design_moment_connection(beam, column, Mu_kip_ft=180.0, Vu_kips=35.0, conn_type="BFP")
    assert result["type"] == "BFP"
    assert "flange_force_kips" in result
    assert "shear_tab" in result
    assert "passes" in result


def test_section_recommender_falls_back_without_model():
    recommendations = predict_section(
        span_ft=30.0,
        dead_load=0.5,
        live_load=0.8,
        beam_spacing=10.0,
        model_path="missing_model.pkl",
        top_k=3,
    )
    assert isinstance(recommendations, list)


def test_process_drawing_writes_raw_json(monkeypatch):
    monkeypatch.setattr("process_drawing_batch.pdf_to_images", lambda *_args, **_kwargs: ["page"])
    monkeypatch.setattr("process_drawing_batch.image_to_base64", lambda *_args, **_kwargs: "b64")
    monkeypatch.setattr("process_drawing_batch.extract_from_image", lambda *_args, **_kwargs: {"beams": [{"mark": "B1"}], "columns": [], "girders": []})
    monkeypatch.setattr("process_drawing_batch.merge_extractions", lambda items: {"beams": [{"mark": "B1"}], "columns": [], "girders": []})

    output_dir = Path(f"C:/Users/harsh/OneDrive/Desktop/Civil Agent, model 1/beam_rl_project/data/test_batch_output_{uuid4().hex}")
    result = process_drawing("example.pdf", output_dir=str(output_dir))
    raw_path = output_dir / "example_raw.json"
    assert raw_path.exists()
    assert result["beams"][0]["mark"] == "B1"
    assert json.loads(raw_path.read_text(encoding="utf-8"))["beams"][0]["mark"] == "B1"


def test_conversational_agent_intent_classifier_without_init():
    agent = ConversationalCivilAgent.__new__(ConversationalCivilAgent)
    agent.current_design = None
    assert agent._classify_intent("design a 30ft beam") == "new_design"
    agent.current_design = {"beam_name": "W14x53"}
    assert agent._classify_intent("what if the span is 25ft?") == "what_if"
    assert agent._classify_intent("why did you choose that beam?") == "explain"
    assert agent._classify_intent("compare the top 3 options") == "compare"
