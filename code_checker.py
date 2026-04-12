"""
Code compliance checker for Civil Agent.

This layer keeps limit-state logic, phi factors, code-equation references, and
fix suggestions separate from the mechanics routines in beam_physics.py.
"""

from __future__ import annotations

from beams_data import BEAMS_DF


PHI_B = 0.90
PHI_V = 1.00
PHI_C = 0.90


def check_moment(
    Mu_kip_ft,
    phi_Mn_kip_ft,
    beam_name,
    Lb_ft,
    Lp_ft,
    Lr_ft,
    ltb_zone="no_ltb",
):
    ratio = Mu_kip_ft / phi_Mn_kip_ft if phi_Mn_kip_ft > 0 else 999.0
    passes = ratio <= 1.0
    note = None
    if not passes:
        note = (
            f"Mu ({Mu_kip_ft:.1f} kip-ft) exceeds phiMn ({phi_Mn_kip_ft:.1f} kip-ft). "
            f"LTB zone: {ltb_zone}. Options: (1) increase section, (2) reduce unbraced length "
            f"below {Lp_ft:.1f}ft to eliminate LTB reduction. Reference: AISC 360-22 Section F2."
        )
    return {
        "check": "moment",
        "passes": passes,
        "ratio": round(ratio, 3),
        "demand_kip_ft": round(Mu_kip_ft, 1),
        "capacity_kip_ft": round(phi_Mn_kip_ft, 1),
        "equation": (
            "AISC 360-22 Eq. F2-1 (phiMp)"
            if Lb_ft <= Lp_ft
            else "AISC 360-22 Eq. F2-2"
            if Lb_ft <= Lr_ft
            else "AISC 360-22 Eq. F2-3"
        ),
        "phi": PHI_B,
        "Lb_ft": round(float(Lb_ft), 2),
        "Lp_ft": round(float(Lp_ft), 2),
        "Lr_ft": round(float(Lr_ft), 2),
        "ltb_zone": ltb_zone,
        "note": note,
    }


def check_ltb_aisc360(
    Mu_kip_ft,
    Lb_ft,
    Lp_ft,
    Lr_ft,
    phi_Mp,
    phi_Mn,
    beam_name,
):
    if Lb_ft <= Lp_ft:
        zone = "no_ltb"
        equation = "AISC 360-22 Eq. F2-1"
    elif Lb_ft <= Lr_ft:
        zone = "inelastic"
        equation = "AISC 360-22 Eq. F2-2"
    else:
        zone = "elastic"
        equation = "AISC 360-22 Eq. F2-3/F2-4"

    ratio = Mu_kip_ft / phi_Mn if phi_Mn > 0 else 999.0
    passes = ratio <= 1.0
    note = None
    if zone == "inelastic":
        note = (
            f"Lb ({Lb_ft:.1f} ft) is in the inelastic LTB zone "
            f"(Lp={Lp_ft:.1f} ft < Lb < Lr={Lr_ft:.1f} ft). "
            f"Capacity is reduced to {0 if phi_Mp <= 0 else (phi_Mn / phi_Mp) * 100:.0f}% of phiMp."
        )
    elif zone == "elastic":
        note = (
            f"Lb ({Lb_ft:.1f} ft) exceeds Lr ({Lr_ft:.1f} ft), placing {beam_name} "
            f"in the elastic LTB zone with reduced flexural capacity."
        )
    if not passes:
        fail_note = (
            f" Mu ({Mu_kip_ft:.1f} kip-ft) exceeds LTB-reduced phiMn ({phi_Mn:.1f} kip-ft). "
            f"Add lateral bracing or upgrade the section."
        )
        note = (note or "") + fail_note

    return {
        "check": "ltb",
        "passes": passes,
        "ratio": round(ratio, 3),
        "zone": zone,
        "Lb_ft": round(float(Lb_ft), 2),
        "Lp_ft": round(float(Lp_ft), 2),
        "Lr_ft": round(float(Lr_ft), 2),
        "phi_Mp_kip_ft": round(float(phi_Mp), 1),
        "capacity_kip_ft": round(float(phi_Mn), 1),
        "equation": equation,
        "note": note.strip() if note else None,
    }


def check_deflection(
    delta_in,
    span_ft,
    limit_divisor,
    load_type="live",
):
    delta_limit_in = span_ft * 12.0 / limit_divisor
    ratio = delta_in / delta_limit_in if delta_limit_in > 0 else 999.0
    passes = ratio <= 1.0
    note = None
    if not passes:
        note = (
            f"{load_type.title()} load deflection ({delta_in:.2f} in) exceeds "
            f"the L/{limit_divisor} limit ({delta_limit_in:.2f} in). "
            f"Increase stiffness or reduce span."
        )
    elif ratio >= 0.9:
        note = (
            f"{load_type.title()} load deflection ({delta_in:.2f} in) approaches "
            f"the L/{limit_divisor} limit ({delta_limit_in:.2f} in)."
        )
    return {
        "check": "deflection",
        "passes": passes,
        "ratio": round(ratio, 3),
        "delta_in": round(float(delta_in), 4),
        "limit_in": round(float(delta_limit_in), 4),
        "limit_str": f"L/{limit_divisor}",
        "equation": "AISC 360-22 Section L3",
        "load_type": load_type,
        "note": note,
    }


def check_shear(
    Vu_kips,
    phi_Vn_kips,
    beam_name,
    tw,
    d,
):
    ratio = Vu_kips / phi_Vn_kips if phi_Vn_kips > 0 else 999.0
    passes = ratio <= 1.0
    note = None
    if not passes:
        note = (
            f"Vu ({Vu_kips:.1f} kips) exceeds phiVn ({phi_Vn_kips:.1f} kips). "
            f"Increase web thickness or choose a heavier section. Reference: AISC 360-22 Section G2."
        )
    return {
        "check": "shear",
        "passes": passes,
        "ratio": round(ratio, 3),
        "demand_kips": round(float(Vu_kips), 1),
        "capacity_kips": round(float(phi_Vn_kips), 1),
        "tw_in": round(float(tw), 3),
        "d_in": round(float(d), 3),
        "equation": "AISC 360-22 Eq. G2-1",
        "phi": PHI_V,
        "note": note,
    }


def check_local_buckling(
    flange_ratio,
    web_ratio,
    flange_limit,
    web_limit,
    beam_name,
):
    flange_check_ratio = flange_ratio / flange_limit if flange_limit > 0 else 0.0
    web_check_ratio = web_ratio / web_limit if web_limit > 0 else 0.0
    ratio = max(flange_check_ratio, web_check_ratio)
    passes = ratio <= 1.0
    note = None
    if not passes:
        controlling = "flange" if flange_check_ratio >= web_check_ratio else "web"
        note = (
            f"{beam_name} fails {controlling} compactness in AISC Table B4.1b. "
            f"Use a stockier section with thicker elements."
        )
    return {
        "check": "local_buckling",
        "passes": passes,
        "ratio": round(ratio, 3),
        "flange_ratio": round(float(flange_ratio), 3),
        "web_ratio": round(float(web_ratio), 3),
        "flange_limit": round(float(flange_limit), 3),
        "web_limit": round(float(web_limit), 3),
        "flange_check_ratio": round(flange_check_ratio, 3),
        "web_check_ratio": round(web_check_ratio, 3),
        "equation": "AISC 360-22 Table B4.1b",
        "note": note,
    }


def suggest_fix(failed_check, beam, span_ft, demands):
    """
    Suggest a specific next step for a failed beam design.
    """
    from beam_physics import evaluate_beam_code

    current_weight = float(beam.get("weight", 0))
    dead_load = float(demands.get("dead_load", 0))
    live_load = float(demands.get("live_load", 0))
    point_load = float(demands.get("point_load", 0))
    Lb_ft = float(demands.get("Lb_ft", span_ft))
    defl_limit = int(demands.get("defl_limit", 360))

    upgrade_text = None
    for _, row in BEAMS_DF.iterrows():
        if float(row["weight"]) <= current_weight:
            continue
        candidate = row.to_dict()
        candidate_result = evaluate_beam_code(
            span_ft,
            dead_load,
            live_load,
            candidate,
            Lb_ft=Lb_ft,
            point_load=point_load,
            defl_limit=defl_limit,
            include_suggestions=False,
        )
        if candidate_result["passes"]:
            upgrade_text = (
                f"Upgrade to {candidate['name']} "
                f"(max utilization {candidate_result['worst_ratio']:.2f})."
            )
            break

    check_name = failed_check.get("check")
    if check_name == "ltb":
        brace_text = None
        lp = failed_check.get("Lp_ft")
        lr = failed_check.get("Lr_ft")
        if lp:
            brace_text = f"Add lateral bracing at about {lp:.1f} ft spacing to eliminate LTB"
        elif lr:
            brace_text = f"Reduce unbraced length to about {lr:.1f} ft or less"
        parts = [part for part in [brace_text, upgrade_text] if part]
        return " or ".join(parts) if parts else "Reduce unbraced length or upgrade the beam."

    if check_name == "deflection":
        if upgrade_text:
            return f"{upgrade_text} For composite options, consider a higher composite ratio if available."
        return "Increase stiffness with a heavier or composite section."

    if upgrade_text:
        return upgrade_text
    return "Review the next heavier section or shorten the span."


def generate_check_report(beam_name, span_ft, dead_load, live_load, checks):
    Wu = 1.2 * dead_load + 1.6 * live_load
    Mu = Wu * span_ft**2 / 8
    Vu = Wu * span_ft / 2
    all_pass = all(check.get("passes", False) for check in checks)
    ratios = {check["check"]: float(check.get("ratio", 0.0)) for check in checks}
    controlling = max(ratios, key=ratios.get) if ratios else "unknown"
    failed_checks = [check["check"] for check in checks if not check.get("passes", False)]
    fix_suggestions = [check.get("note") for check in checks if not check.get("passes", False) and check.get("note")]
    return {
        "member": beam_name,
        "span_ft": span_ft,
        "loads": {
            "dead_kip_ft": round(dead_load, 3),
            "live_kip_ft": round(live_load, 3),
            "Wu_kip_ft": round(Wu, 3),
            "Mu_kip_ft": round(Mu, 1),
            "Vu_kips": round(Vu, 1),
        },
        "checks": checks,
        "overall": "PASS" if all_pass else "FAIL",
        "controlling_check": controlling,
        "controlling_ratio": round(ratios.get(controlling, 0.0), 3),
        "failed_checks": failed_checks,
        "fix_suggestions": fix_suggestions,
        "code_edition": "AISC 360-22",
    }


def generate_full_report(beam_name, span_ft, dead_load, live_load, all_check_results):
    checks = list(all_check_results.get("checks", []))
    controlling = max(checks, key=lambda item: float(item.get("ratio", 0))) if checks else {}
    overall = "PASS" if all(check.get("passes", False) for check in checks) else "FAIL"

    fix_suggestion = None
    if overall == "FAIL" and all_check_results.get("include_suggestions", True):
        beam = all_check_results.get("beam", {})
        fix_suggestion = suggest_fix(
            controlling,
            beam,
            span_ft,
            all_check_results.get("demands", {}),
        )
        if fix_suggestion and not controlling.get("note"):
            controlling["note"] = fix_suggestion

    return {
        "member": beam_name,
        "span_ft": round(float(span_ft), 2),
        "loads": {
            "dead": round(float(dead_load), 3),
            "live": round(float(live_load), 3),
            "Wu": round(float(all_check_results.get("demands", {}).get("Wu", 0)), 3),
        },
        "demands": {
            "Mu": round(float(all_check_results.get("demands", {}).get("Mu", 0)), 2),
            "Vu": round(float(all_check_results.get("demands", {}).get("Vu", 0)), 2),
        },
        "checks": checks,
        "overall": overall,
        "controlling_check": controlling.get("check", "unknown"),
        "controlling_ratio": round(float(controlling.get("ratio", 0)), 3),
        "code_edition": "AISC 360-22",
        "fix_suggestion": fix_suggestion,
    }


def check_moment_aisc360(*args, **kwargs):
    return check_moment(*args, **kwargs)


def check_shear_aisc360(*args, **kwargs):
    return check_shear(*args, **kwargs)


def check_deflection_aisc360(*args, **kwargs):
    return check_deflection(*args, **kwargs)


def check_local_buckling_aisc360(*args, **kwargs):
    return check_local_buckling(*args, **kwargs)
