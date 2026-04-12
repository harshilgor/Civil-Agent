"""
Integration wrapper for the standalone ACI 318 concrete engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parent
_PARENT = _ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from engine import CodeEdition, ConcreteSection, MemberForces, RebarLayer, Stirrups  # noqa: E402
from engine.aci318 import check_flexure, check_shear  # noqa: E402


def run_rc_beam_checks(
    *,
    b_in: float,
    h_in: float,
    d_in: float,
    fc_psi: float,
    fy_psi: float,
    tension_bar_count: int,
    tension_bar_size: int,
    Mu_kip_ft: float,
    Vu_kips: float,
    stirrup_bar_size: int = 0,
    stirrup_spacing_in: float = 0.0,
    stirrup_legs: int = 2,
    use_detailed_shear: bool = False,
) -> dict[str, Any]:
    stirrups = None
    if stirrup_bar_size > 0 and stirrup_spacing_in > 0:
        stirrups = Stirrups(
            bar_size=int(stirrup_bar_size),
            spacing=float(stirrup_spacing_in),
            legs=int(stirrup_legs),
        )

    section = ConcreteSection(
        b=float(b_in),
        h=float(h_in),
        d=float(d_in),
        fc=float(fc_psi),
        fy=float(fy_psi),
        rebar_tension=RebarLayer(count=int(tension_bar_count), bar_size=int(tension_bar_size)),
        stirrups=stirrups,
    )
    forces = MemberForces.from_kips(
        Mu_kip_ft=float(Mu_kip_ft),
        Vu_kips=float(Vu_kips),
        combo_name="Strength",
    )

    flexure = check_flexure(section, forces, code=CodeEdition.ACI318_19)
    shear = check_shear(
        section,
        forces,
        code=CodeEdition.ACI318_19,
        use_detailed=bool(use_detailed_shear),
    )

    checks = [flexure, shear]
    controlling = max(checks, key=lambda item: float(item.dcr))
    overall_status = "PASS"
    if any(item.status.value == "fail" for item in checks):
        overall_status = "FAIL"
    elif any(item.status.value == "warning" for item in checks):
        overall_status = "WARNING"

    return {
        "section": section,
        "forces": forces,
        "inputs": {
            "b_in": float(b_in),
            "h_in": float(h_in),
            "d_in": float(d_in),
            "fc_psi": float(fc_psi),
            "fy_psi": float(fy_psi),
            "tension_bar_count": int(tension_bar_count),
            "tension_bar_size": int(tension_bar_size),
            "stirrup_bar_size": int(stirrup_bar_size),
            "stirrup_spacing_in": float(stirrup_spacing_in),
            "stirrup_legs": int(stirrup_legs),
            "Mu_kip_ft": float(Mu_kip_ft),
            "Vu_kips": float(Vu_kips),
            "use_detailed_shear": bool(use_detailed_shear),
        },
        "flexure": flexure,
        "shear": shear,
        "controlling_check": controlling.check_type.value,
        "controlling_dcr": float(controlling.dcr),
        "overall_status": overall_status,
        "warnings": [*flexure.warnings, *shear.warnings],
    }
