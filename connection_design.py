"""
Shear tab connection design (beam web to column flange).
A325-N bolts (3/4\", 7/8\", 1\"), A36 plate, 3/16\" fillet welds both sides.
"""

from __future__ import annotations

import math

BOLT_SIZES = [
    {"dia": 0.75, "phi_single": 15.9, "label": '3/4"'},
    {"dia": 0.875, "phi_single": 21.6, "label": '7/8"'},
    {"dia": 1.0, "phi_single": 28.3, "label": '1"'},
]

BOLT_SPACING = 3.0
EDGE_DISTANCE = 1.5
TAB_FY = 36  # ksi, A36
TAB_FU = 58  # ksi
PHI_SHEAR = 1.0
PHI_BEARING = 0.75
WELD_SIZE = 0.1875  # 3/16" fillet
FU_BEAM = 65  # ksi, typical web material for bearing

FAILURE_MESSAGE = (
    "Shear demand exceeds standard bolt capacity.\n"
    "Consider welded connection or moment frame."
)

MOMENT_BOLT_TENSION_7_8 = 28.3  # kips
PLATE_FY = 50  # ksi
WELD_FEXX = 70  # ksi


def check_bolt_shear(Vu_kips, n_bolts, phi_single):
    """phi*Rn = n * phi_single (kips per bolt, single shear)."""
    phi_Rn = n_bolts * phi_single
    ratio = Vu_kips / phi_Rn
    return ratio <= 1.0, round(ratio, 3)


def check_bolt_bearing(Vu_kips, n_bolts, tw, bolt_dia):
    """phi*rn per bolt = PHI_BEARING * 2.4 * 65 * bolt_dia * tw"""
    phi_rn = PHI_BEARING * 2.4 * FU_BEAM * bolt_dia * tw
    phi_Rn = n_bolts * phi_rn
    ratio = Vu_kips / phi_Rn
    return ratio <= 1.0, round(ratio, 3)


def check_tab_shear(Vu_kips, tab_thickness, tab_length):
    """Gross shear yielding: phi*Vn = 1.0 * 0.6 * Fy * (t * L)"""
    Ag = tab_thickness * tab_length
    phi_Vn = PHI_SHEAR * 0.6 * TAB_FY * Ag
    ratio = Vu_kips / phi_Vn
    return ratio <= 1.0, round(ratio, 3)


def check_weld(Vu_kips, tab_length):
    """3/16\" fillet both sides along tab length."""
    weld_per_inch = 2 * 0.707 * WELD_SIZE * 0.6 * 70
    phi_Rn = weld_per_inch * tab_length
    ratio = Vu_kips / phi_Rn
    return ratio <= 1.0, round(ratio, 3)


def _tab_length_inches(n_bolts):
    """Single vertical row: (n-1) spaces + 2 end distances."""
    return (n_bolts - 1) * BOLT_SPACING + 2 * EDGE_DISTANCE


def _fmt_thickness_in(t):
    """Common plate thicknesses as fractions for display."""
    if abs(t - 0.375) < 0.001:
        return '3/8"'
    if abs(t - 0.5) < 0.001:
        return '1/2"'
    if abs(t - 0.625) < 0.001:
        return '5/8"'
    return f'{t:g}"'


def ceil_to_nearest(value, increment):
    return math.ceil(float(value) / float(increment)) * float(increment)


def _check_panel_zone(column, Pf_kips):
    dc = float(column.get("d", 14.0))
    tw = float(column.get("tw", 0.5))
    bcf = float(column.get("bf", 14.0))
    tcf = float(column.get("tf", 0.75))
    db = max(float(column.get("d", dc)), 1.0)
    Rn = 0.60 * TAB_FY * dc * tw * (1.0 + 3.0 * bcf * tcf**2 / max(db * dc * tw, 1e-6))
    phi_Rn = 1.0 * Rn
    return Pf_kips <= phi_Rn


def _design_bfp(beam, column, Pf_kips, Vu_kips):
    bf_beam = float(beam["bf"])
    bp = max(bf_beam, 7.0)
    tp_req = Pf_kips / max(0.90 * PLATE_FY * bp, 1e-6)
    tp = ceil_to_nearest(max(tp_req, 0.5), 0.125)
    n_bolts = max(4, int(Pf_kips / max(0.90 * MOMENT_BOLT_TENSION_7_8, 1e-6)) + 1)
    bearing_cap = n_bolts * 0.90 * 2.4 * TAB_FU * 0.875 * tp
    bearing_ok = Pf_kips <= bearing_cap
    pz_ok = _check_panel_zone(column, Pf_kips)
    weld_len = 2.0 * bp
    w_req = Pf_kips / max(0.707 * 0.6 * WELD_FEXX * weld_len, 1e-6)
    w_actual = ceil_to_nearest(max(w_req, 0.25), 0.125)
    passes = bearing_ok and pz_ok
    cost = tp * bp * 12.0 * 0.2833 * 1.50 * 2.0 + n_bolts * 2.0 * 25.0 + weld_len * 12.0 * w_actual * 15.0
    return {
        "type": "BFP",
        "Mu_capacity_kip_ft": round(Pf_kips * max(float(beam["d"]) - float(beam["tf"]), 1.0) / 12.0, 1),
        "Vu_capacity_kips": round(max(Vu_kips, 0.0) * 1.10, 1),
        "flange_force_kips": round(Pf_kips, 1),
        "plate_thickness_in": round(tp, 3),
        "plate_width_in": round(bp, 1),
        "num_bolts_flange": int(n_bolts),
        "bolt_size": '7/8" A325',
        "weld_size_in": round(w_actual, 3),
        "panel_zone_ok": pz_ok,
        "passes": passes,
        "cost_estimate": round(cost, 0),
        "details": {
            "bearing_ok": bearing_ok,
            "bearing_capacity_kips": round(bearing_cap, 1),
        },
        "shear_tab": design_shear_tab(Vu_kips, beam),
    }


def _design_eep(beam, column, Pf_kips, Vu_kips):
    bp = max(float(beam["bf"]) + 2.0, 8.0)
    tp = ceil_to_nearest(max(Pf_kips / max(0.90 * PLATE_FY * bp, 1e-6), 0.75), 0.125)
    n_bolts = max(8, int(Pf_kips / max(0.90 * MOMENT_BOLT_TENSION_7_8, 1e-6)) + 2)
    pz_ok = _check_panel_zone(column, Pf_kips)
    cost = tp * bp * (float(beam["d"]) + 6.0) * 0.2833 * 1.60 + n_bolts * 35.0 + 450.0
    return {
        "type": "EEP",
        "Mu_capacity_kip_ft": round(Pf_kips * max(float(beam["d"]) - float(beam["tf"]), 1.0) / 12.0, 1),
        "Vu_capacity_kips": round(max(Vu_kips, 0.0) * 1.10, 1),
        "flange_force_kips": round(Pf_kips, 1),
        "plate_thickness_in": round(tp, 3),
        "plate_width_in": round(bp, 1),
        "num_bolts_flange": int(n_bolts),
        "bolt_size": '7/8" A325',
        "weld_size_in": 0.375,
        "panel_zone_ok": pz_ok,
        "passes": pz_ok,
        "cost_estimate": round(cost, 0),
        "details": {"connection_style": "Extended end plate"},
        "shear_tab": design_shear_tab(Vu_kips, beam),
    }


def _design_wuf(beam, column, Pf_kips, Vu_kips):
    weld_size = ceil_to_nearest(max(Pf_kips / max(0.707 * 0.6 * WELD_FEXX * max(float(beam["bf"]) * 2.0, 1.0), 1e-6), 0.3125), 0.0625)
    pz_ok = _check_panel_zone(column, Pf_kips)
    cost = max(float(beam["bf"]), 8.0) * max(float(beam["d"]), 12.0) * weld_size * 18.0 + 300.0
    return {
        "type": "WUF-W",
        "Mu_capacity_kip_ft": round(Pf_kips * max(float(beam["d"]) - float(beam["tf"]), 1.0) / 12.0, 1),
        "Vu_capacity_kips": round(max(Vu_kips, 0.0) * 1.05, 1),
        "flange_force_kips": round(Pf_kips, 1),
        "plate_thickness_in": 0.0,
        "plate_width_in": round(float(beam["bf"]), 1),
        "num_bolts_flange": 0,
        "bolt_size": "N/A",
        "weld_size_in": round(weld_size, 3),
        "panel_zone_ok": pz_ok,
        "passes": pz_ok,
        "cost_estimate": round(cost, 0),
        "details": {"connection_style": "Welded unreinforced flange"},
        "shear_tab": design_shear_tab(Vu_kips, beam),
    }


def design_moment_connection(beam, column, Mu_kip_ft, Vu_kips, conn_type="BFP"):
    """
    Preliminary prequalified steel moment-connection sizing.
    """
    Mu_in = float(Mu_kip_ft) * 12.0
    d_beam = float(beam["d"])
    tf_beam = float(beam["tf"])
    Pf = Mu_in / max(d_beam - tf_beam, 1e-6)

    conn_key = str(conn_type).upper()
    if conn_key == "BFP":
        return _design_bfp(beam, column, Pf, Vu_kips)
    if conn_key == "EEP":
        return _design_eep(beam, column, Pf, Vu_kips)
    if conn_key == "WUF-W":
        return _design_wuf(beam, column, Pf, Vu_kips)
    raise ValueError(f"Unsupported moment connection type '{conn_type}'.")


def explain_connection(result):
    """Plain English summary; includes bolt size and tab size."""
    n = result["num_bolts"]
    t = result["tab_thickness"]
    L = result["tab_length"]
    bolt_label = result.get("bolt_label", '3/4"')
    chk = result["checks"]
    controlling = max(chk, key=chk.get)
    labels = {
        "bolt_shear": "bolt shear",
        "bolt_bearing": "bolt bearing",
        "tab_shear": "tab shear",
        "weld": "weld capacity",
    }
    ctrl_name = labels.get(controlling, controlling)
    ctrl_val = chk[controlling]

    if not result.get("passes", False) and result.get("failure_reason"):
        return result["failure_reason"]

    lines = [
        f'{n} - {bolt_label} A325-N bolts, {_fmt_thickness_in(t)} x {L:g}" shear tab',
        f"Spacing: {BOLT_SPACING:g} in.  |  Weld: 3/16 in. fillet both sides to column flange",
        f"Controlling check: {ctrl_name} at {ctrl_val:.2f}",
    ]
    if not result.get("passes", False):
        lines.insert(
            0,
            "WARNING: Connection could not satisfy all limits within "
            f"{result.get('max_bolts', 7)} bolts / plate thicknesses / bolt sizes tried. ",
        )
    return "\n".join(lines)


def design_shear_tab(Vu_kips, beam):
    """
    Try 3/4\", then 7/8\", then 1\" bolts; n = 2-7; plate 3/8\", 1/2\", 5/8\".
    """
    tw = float(beam["tw"])
    max_bolts = 7
    thicknesses = (0.375, 0.5, 0.625)

    last = None

    for bolt_spec in BOLT_SIZES:
        dia = bolt_spec["dia"]
        phi_single = bolt_spec["phi_single"]
        bolt_label = bolt_spec["label"]

        for n_bolts in range(2, max_bolts + 1):
            tab_length = _tab_length_inches(n_bolts)
            for tab_thickness in thicknesses:
                ok_bs, r_bs = check_bolt_shear(Vu_kips, n_bolts, phi_single)
                ok_bb, r_bb = check_bolt_bearing(Vu_kips, n_bolts, tw, dia)
                ok_ts, r_ts = check_tab_shear(Vu_kips, tab_thickness, tab_length)
                ok_w, r_w = check_weld(Vu_kips, tab_length)
                checks = {
                    "bolt_shear": r_bs,
                    "bolt_bearing": r_bb,
                    "tab_shear": r_ts,
                    "weld": r_w,
                }
                last = {
                    "num_bolts": n_bolts,
                    "tab_thickness": tab_thickness,
                    "tab_length": round(tab_length, 3),
                    "passes": all([ok_bs, ok_bb, ok_ts, ok_w]),
                    "checks": checks,
                    "max_bolts": max_bolts,
                    "bolt_dia": dia,
                    "bolt_label": bolt_label,
                    "phi_single": phi_single,
                }
                if last["passes"]:
                    last["explanation"] = explain_connection(last)
                    return last

    if last is None:
        tab_length = _tab_length_inches(2)
        dia = BOLT_SIZES[0]["dia"]
        phi_single = BOLT_SIZES[0]["phi_single"]
        ok_bs, r_bs = check_bolt_shear(Vu_kips, 2, phi_single)
        ok_bb, r_bb = check_bolt_bearing(Vu_kips, 2, tw, dia)
        ok_ts, r_ts = check_tab_shear(Vu_kips, 0.375, tab_length)
        ok_w, r_w = check_weld(Vu_kips, tab_length)
        last = {
            "num_bolts": 2,
            "tab_thickness": 0.375,
            "tab_length": round(tab_length, 3),
            "passes": False,
            "checks": {
                "bolt_shear": r_bs,
                "bolt_bearing": r_bb,
                "tab_shear": r_ts,
                "weld": r_w,
            },
            "max_bolts": max_bolts,
            "bolt_dia": dia,
            "bolt_label": BOLT_SIZES[0]["label"],
            "phi_single": phi_single,
        }
    else:
        last["passes"] = False

    last["failure_reason"] = FAILURE_MESSAGE
    last["explanation"] = explain_connection(last)
    return last


if __name__ == "__main__":
    from beams_data import BEAMS_DF

    tests = [
        ("W16x31", 25, "light beam"),
        ("W21x50", 65, "medium beam"),
        ("W36x135", 180, "heavy beam"),
        ("W8x18", 10, "very light"),
    ]

    print("Shear tab connection design - test cases")
    print("=" * 60)
    for beam_name, Vu, label in tests:
        row = BEAMS_DF[BEAMS_DF["name"] == beam_name]
        if row.empty:
            print(f"\n{label}: {beam_name} not in database")
            continue
        beam = row.iloc[0].to_dict()
        result = design_shear_tab(Vu, beam)
        print(f"\n{label}: {beam_name}, Vu={Vu} kips")
        print(f"  Bolts:  {result['num_bolts']} - {result.get('bolt_label', '?')} A325-N")
        print(f"  Tab:    {result['tab_thickness']} in. x {result['tab_length']} in.")
        print(f"  Passes: {result['passes']}")
        print(f"  Checks: {result['checks']}")
        print(f"  {result['explanation']}")
