from __future__ import annotations

from asce7_loads import auto_load_brief
from beams_data import COLUMN_SECTIONS
from lateral_system import compare_systems


def test_auto_load_brief_generates_complete_load_summary(monkeypatch):
    def fake_usgs(*_args, **_kwargs):
        return {
            "request": {"status": "success", "url": "https://example.test/usgs"},
            "response": {
                "data": {
                    "sds": 0.82,
                    "sd1": 0.41,
                    "ss": 1.10,
                    "s1": 0.52,
                    "sdc": "D",
                }
            },
        }

    monkeypatch.setattr("asce7_loads._fetch_usgs_seismic", fake_usgs)

    brief = auto_load_brief(
        city="Chicago",
        num_floors=8,
        floor_height_ft=14.0,
        bay_length_ft=30.0,
        bay_width_ft=30.0,
        occupancy="office",
    )

    assert brief["loads_auto_calculated"] is True
    assert brief["dead_psf"] == 50.0
    assert brief["live_psf"] == 50.0
    assert brief["wind_psf"] > 0
    assert brief["seismic_V_kips"] > 0
    assert len(brief["seismic_Fx"]) == 8
    assert brief["governing_combo"]
    assert brief["seismic_error"] is None


def test_auto_load_brief_gracefully_handles_seismic_lookup_failure(monkeypatch):
    def fail_usgs(*_args, **_kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr("asce7_loads._fetch_usgs_seismic", fail_usgs)

    brief = auto_load_brief(
        city="Chicago",
        num_floors=4,
        floor_height_ft=14.0,
        bay_length_ft=30.0,
        bay_width_ft=25.0,
        occupancy="office",
    )

    assert brief["loads_auto_calculated"] is True
    assert brief["seismic_V_kips"] == 0.0
    assert len(brief["seismic_Fx"]) == 4
    assert "service unavailable" in (brief["seismic_error"] or "")


def test_compare_systems_derives_wind_shear_and_normalizes_column_sections():
    column_name = COLUMN_SECTIONS.iloc[0]["name"]
    building_data = {
        "num_floors": 4,
        "floor_height_ft": 14.0,
        "bay_length_ft": 30.0,
        "bay_width_ft": 25.0,
        "bays_x": 3,
        "bays_y": 2,
        "building_length_ft": 90.0,
        "building_width_ft": 50.0,
    }
    wind_loads = {
        "floor_pressures": [
            {"floor": 1, "pressure_psf": 18.0},
            {"floor": 2, "pressure_psf": 20.0},
            {"floor": 3, "pressure_psf": 22.0},
            {"floor": 4, "pressure_psf": 24.0},
        ]
    }
    seismic_loads = {
        "base_shear_kips": 20.0,
        "floor_forces": [
            {"floor": 1, "force_kips": 5.0},
            {"floor": 2, "force_kips": 5.0},
            {"floor": 3, "force_kips": 5.0},
            {"floor": 4, "force_kips": 5.0},
        ],
    }
    column_sections = [
        {"name": column_name, "Pu_total": 120.0},
        {"name": column_name, "Pu_total": 100.0},
        {"name": column_name, "Pu_total": 80.0},
        {"name": column_name, "Pu_total": 60.0},
    ]

    comparison = compare_systems(
        building_data=building_data,
        wind_loads=wind_loads,
        seismic_loads=seismic_loads,
        column_sections=column_sections,
        architectural_constraints={"open_perimeter": True, "max_brace_bays": 2},
    )

    assert comparison["V_wind_kips"] > comparison["V_seismic_kips"]
    assert comparison["governing_load"] == "wind"
    assert comparison["moment_frame"]["column_schedule"]
    assert comparison["braced_frame"]["brace_schedule"]
    assert comparison["recommendation"] in {"moment_frame", "braced_frame", "shear_wall"}
    assert comparison["reasoning"]
