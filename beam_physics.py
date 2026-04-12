"""
Beam mechanics layer plus compatibility wrapper.
"""

from __future__ import annotations

import math

from beams_data import BEAMS_DF, STEEL_E, STEEL_FY


def apply_lrfd(dead_load, live_load):
    """
    ASCE 7 governing gravity combination used by the current beam workflow.
    """
    return 1.2 * dead_load + 1.6 * live_load


def calculate_demands(span_ft, dead_load, live_load, point_load=0):
    """
    Calculate factored uniform-load demands.
    """
    Wu = apply_lrfd(dead_load, live_load)
    Mu = Wu * span_ft**2 / 8
    Vu = Wu * span_ft / 2

    if point_load > 0:
        Pu = 1.2 * point_load
        Mu += Pu * span_ft / 4
        Vu += Pu / 2

    return Mu, Vu, Wu


def calculate_service_load(dead_load, live_load):
    return dead_load + live_load


def calculate_moment_capacity(beam, Fy=STEEL_FY):
    """
    Raw plastic flexural strength in kip-ft.
    """
    return float(beam["Zx"]) * Fy / 12.0


def calculate_shear_capacity(beam, Fy=STEEL_FY):
    """
    Raw shear strength in kips.
    """
    return 0.6 * Fy * float(beam["d"]) * float(beam["tw"])


def calculate_deflection(span_ft, load_kip_ft, beam, E=STEEL_E, defl_limit=360):
    """
    Actual service-load deflection and allowable deflection in inches.
    """
    L_in = span_ft * 12.0
    w_in = load_kip_ft / 12.0
    delta_in = (5.0 * w_in * L_in**4) / (384.0 * E * float(beam["Ix"]))
    limit_in = L_in / defl_limit
    return {
        "delta_in": delta_in,
        "limit_in": limit_in,
        "limit_divisor": defl_limit,
        "load_type": "live",
    }


def calculate_local_buckling_limits(beam, Fy=STEEL_FY, E=STEEL_E):
    """
    Raw flange/web slenderness and compactness limits.
    """
    flange_limit = 0.38 * (E / Fy) ** 0.5
    web_limit = 3.76 * (E / Fy) ** 0.5

    h_clear = float(beam["d"]) - 2 * float(beam["tf"]) - 2 * float(beam["r"])
    flange_ratio = (float(beam["bf"]) / 2.0) / float(beam["tf"])
    web_ratio = h_clear / float(beam["tw"])

    return {
        "flange_ratio": flange_ratio,
        "web_ratio": web_ratio,
        "flange_limit": flange_limit,
        "web_limit": web_limit,
        "h_clear": h_clear,
    }


def calculate_ltb_properties(beam, Lb_ft, Fy=STEEL_FY, E=STEEL_E):
    """
    Raw LTB geometry and nominal moment capacity.
    """
    ry = float(beam["ry"])
    Zx = float(beam["Zx"])
    Sx = float(beam["Sx"])
    d = float(beam["d"])
    bf = float(beam["bf"])
    tf = float(beam["tf"])
    tw = float(beam["tw"])
    Iy = float(beam["Iy"])

    ho = d - tf
    J = (2.0 * bf * tf**3 + (d - 2.0 * tf) * tw**3) / 3.0
    rts = (Iy * ho / (2.0 * Sx)) ** 0.5
    c = 1.0
    Cb = 1.0

    Mp_kip_ft = Fy * Zx / 12.0
    Lb_in = float(Lb_ft) * 12.0
    Lp_in = 1.76 * ry * (E / Fy) ** 0.5

    jc_sxho = J * c / (Sx * ho)
    Lr_in = 1.95 * rts * (E / (0.7 * Fy)) * math.sqrt(
        jc_sxho + math.sqrt(jc_sxho**2 + 6.76 * (0.7 * Fy / E) ** 2)
    )

    if Lb_in <= Lp_in:
        zone = "no_ltb"
        Mn_kip_ft = Mp_kip_ft
    elif Lb_in <= Lr_in:
        zone = "inelastic"
        Mn_kip_ft = (
            Mp_kip_ft
            - (Mp_kip_ft - (0.7 * Fy * Sx / 12.0)) * (Lb_in - Lp_in) / (Lr_in - Lp_in)
        )
        Mn_kip_ft = min(Mn_kip_ft, Mp_kip_ft)
    else:
        zone = "elastic"
        Fcr = (Cb * math.pi**2 * E / (Lb_in / rts) ** 2) * math.sqrt(
            1.0 + 0.078 * jc_sxho * (Lb_in / rts) ** 2
        )
        Mn_kip_ft = min(Fcr * Sx / 12.0, Mp_kip_ft)

    return {
        "Lb_ft": float(Lb_ft),
        "Lp_ft": Lp_in / 12.0,
        "Lr_ft": Lr_in / 12.0,
        "zone": zone,
        "Mp_kip_ft": Mp_kip_ft,
        "Mn_kip_ft": Mn_kip_ft,
        "J": J,
        "rts": rts,
        "ho": ho,
    }


def evaluate_beam_code(
    span_ft,
    dead_load,
    live_load,
    beam,
    Lb_ft=None,
    point_load=0,
    defl_limit=360,
    Fy=STEEL_FY,
    include_suggestions=True,
):
    """
    Compatibility-safe beam evaluation using pure mechanics + code checker.
    """
    from code_checker import (
        PHI_B,
        PHI_V,
        check_deflection,
        check_local_buckling,
        check_ltb_aisc360,
        check_moment,
        check_shear,
        generate_full_report,
    )

    if Lb_ft is None:
        Lb_ft = span_ft

    beam_name = beam.get("name", "Unknown")
    Mu, Vu, Wu = calculate_demands(span_ft, dead_load, live_load, point_load)
    service_load = calculate_service_load(dead_load, live_load)
    deflection = calculate_deflection(span_ft, service_load, beam, defl_limit=defl_limit)
    local_buckling = calculate_local_buckling_limits(beam, Fy=Fy)
    ltb = calculate_ltb_properties(beam, Lb_ft, Fy=Fy)

    Mn_kip_ft = calculate_moment_capacity(beam, Fy=Fy)
    Vn_kips = calculate_shear_capacity(beam, Fy=Fy)
    phi_Mn_kip_ft = PHI_B * Mn_kip_ft
    phi_Vn_kips = PHI_V * Vn_kips
    phi_Mp_kip_ft = PHI_B * ltb["Mp_kip_ft"]
    phi_Mn_ltb_kip_ft = PHI_B * ltb["Mn_kip_ft"]

    moment_check = check_moment(
        Mu,
        phi_Mn_kip_ft,
        beam_name,
        ltb["Lb_ft"],
        ltb["Lp_ft"],
        ltb["Lr_ft"],
        ltb["zone"],
    )
    shear_check = check_shear(
        Vu,
        phi_Vn_kips,
        beam_name,
        float(beam["tw"]),
        float(beam["d"]),
    )
    deflection_check = check_deflection(
        deflection["delta_in"],
        span_ft,
        deflection["limit_divisor"],
        load_type=deflection["load_type"],
    )
    local_buckling_check = check_local_buckling(
        local_buckling["flange_ratio"],
        local_buckling["web_ratio"],
        local_buckling["flange_limit"],
        local_buckling["web_limit"],
        beam_name,
    )
    ltb_check = check_ltb_aisc360(
        Mu,
        ltb["Lb_ft"],
        ltb["Lp_ft"],
        ltb["Lr_ft"],
        phi_Mp_kip_ft,
        phi_Mn_ltb_kip_ft,
        beam_name,
    )

    all_check_results = {
        "beam": beam,
        "checks": [
            moment_check,
            ltb_check,
            shear_check,
            deflection_check,
            local_buckling_check,
        ],
        "demands": {
            "Mu": Mu,
            "Vu": Vu,
            "Wu": Wu,
            "service_load": service_load,
            "dead_load": dead_load,
            "live_load": live_load,
            "point_load": point_load,
            "Lb_ft": Lb_ft,
            "defl_limit": defl_limit,
        },
        "capacities": {
            "Mn_kip_ft": Mn_kip_ft,
            "phi_Mn_kip_ft": phi_Mn_kip_ft,
            "Vn_kips": Vn_kips,
            "phi_Vn_kips": phi_Vn_kips,
            "Mp_kip_ft": ltb["Mp_kip_ft"],
            "phi_Mp_kip_ft": phi_Mp_kip_ft,
            "Mn_ltb_kip_ft": ltb["Mn_kip_ft"],
            "phi_Mn_ltb_kip_ft": phi_Mn_ltb_kip_ft,
        },
        "ltb": ltb,
        "deflection": deflection,
        "local_buckling": local_buckling,
        "include_suggestions": include_suggestions,
    }
    full_report = generate_full_report(
        beam_name,
        span_ft,
        dead_load,
        live_load,
        all_check_results,
    )

    code_checks = {check["check"]: check for check in full_report["checks"]}
    details = {
        "moment_ratio": round(float(code_checks["moment"]["ratio"]), 3),
        "shear_ratio": round(float(code_checks["shear"]["ratio"]), 3),
        "defl_ratio": round(float(code_checks["deflection"]["ratio"]), 3),
        "flange_ratio": round(float(code_checks["local_buckling"]["flange_check_ratio"]), 3),
        "web_ratio": round(float(code_checks["local_buckling"]["web_check_ratio"]), 3),
        "ltb_ratio": round(float(code_checks["ltb"]["ratio"]), 3),
        "ltb_zone": str(code_checks["ltb"]["zone"]).replace("_", " "),
        "code_checks": code_checks,
        "full_report": full_report,
        "controlling_check": full_report["controlling_check"],
        "fix_suggestion": full_report.get("fix_suggestion"),
        "equation": code_checks[full_report["controlling_check"]]["equation"],
    }

    numeric_ratios = [
        details["moment_ratio"],
        details["shear_ratio"],
        details["defl_ratio"],
        details["flange_ratio"],
        details["web_ratio"],
        details["ltb_ratio"],
    ]

    return {
        "passes": full_report["overall"] == "PASS",
        "weight": float(beam["weight"]),
        "worst_ratio": max(numeric_ratios),
        "details": details,
        "full_report": full_report,
    }


def check_beam_design(
    span_ft,
    dead_load,
    live_load,
    beam,
    Lb_ft=None,
    point_load=0,
    defl_limit=360,
    Fy=STEEL_FY,
):
    """
    Stable compatibility wrapper used across the app.
    """
    result = evaluate_beam_code(
        span_ft,
        dead_load,
        live_load,
        beam,
        Lb_ft=Lb_ft,
        point_load=point_load,
        defl_limit=defl_limit,
        Fy=Fy,
        include_suggestions=True,
    )
    return result["passes"], result["weight"], result["worst_ratio"], result["details"]


def explain_result(beam_name, passes, details, alternatives=None):
    """
    Richer explanation using the compliance report when present.
    """
    full_report = details.get("full_report", {})
    if full_report:
        controlling = full_report.get("controlling_check", "unknown")
        checks = {check["check"]: check for check in full_report.get("checks", [])}
        controlling_check = checks.get(controlling, {})
        equation = controlling_check.get("equation", "AISC 360-22")
        ratio = float(full_report.get("controlling_ratio", details.get("moment_ratio", 0)))
        if passes:
            explanation = (
                f"{beam_name} passes all checks. "
                f"Controlling check: {controlling.replace('_', ' ')} "
                f"at {ratio:.2f} utilization ({equation})."
            )
        else:
            note = controlling_check.get("note") or full_report.get("fix_suggestion") or "Review section selection."
            explanation = (
                f"{beam_name} fails {controlling.replace('_', ' ')} "
                f"at {ratio:.2f} utilization ({equation}). {note}"
            )
    else:
        check_names = {
            "moment_ratio": "moment capacity",
            "shear_ratio": "shear capacity",
            "defl_ratio": "deflection",
            "flange_ratio": "flange local buckling",
            "web_ratio": "web local buckling",
            "ltb_ratio": "lateral torsional buckling",
        }
        if not passes:
            failed = {
                k: v for k, v in details.items()
                if isinstance(v, (int, float)) and v > 1.0
            }
            worst_key = max(failed, key=failed.get)
            worst_name = check_names.get(worst_key, worst_key)
            explanation = (
                f"{beam_name} fails {worst_name} "
                f"(utilization {failed[worst_key]:.2f} -- must be below 1.0)."
            )
        else:
            numeric = {k: v for k, v in details.items() if isinstance(v, (int, float))}
            controlling_key = max(numeric, key=numeric.get)
            controlling_name = check_names.get(controlling_key, controlling_key)
            controlling_val = numeric[controlling_key]
            explanation = (
                f"{beam_name} passes all checks. "
                f"Controlling check: {controlling_name} "
                f"at {controlling_val:.2f} utilization."
            )

    if details.get("fix_suggestion") and not passes and not full_report:
        explanation += f" Suggested fix: {details['fix_suggestion']}"

    if alternatives:
        explanation += "\n\nAlternatives considered:"
        for alt_name, alt_passes, alt_weight in alternatives:
            status = "passes" if alt_passes else "fails"
            explanation += f"\n  {alt_name} ({alt_weight:.0f} lb/ft) -- {status}"

    return explanation


if __name__ == "__main__":
    tests = [
        ("W8x31", 15, 0.4, 0.6, None),
        ("W8x31", 30, 2.0, 3.0, None),
        ("W36x300", 40, 1.6, 2.4, None),
        ("W14x22", 20, 0.5, 1.0, 10),
    ]

    print("Beam Mechanics + Code Checker Smoke Test")
    print("=" * 60)
    for beam_name, span, dl, ll, lb in tests:
        row = BEAMS_DF[BEAMS_DF["name"] == beam_name]
        if row.empty:
            print(f"{beam_name} not found in database")
            continue
        beam = row.iloc[0].to_dict()
        passes, weight, worst, details = check_beam_design(span, dl, ll, beam, Lb_ft=lb)
        print(f"\n{beam_name} | {span}ft | DL={dl} LL={ll} | Lb={lb}")
        print(f"  Passes: {passes}  Weight: {weight:.1f} lb/ft")
        print(f"  Worst ratio: {worst:.3f}")
        print(f"  Explanation: {explain_result(beam_name, passes, details)}")
