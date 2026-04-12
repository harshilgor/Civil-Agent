"""
Column Physics Engine
AISC 360-22 steel column checks: axial, P-M interaction, local buckling
"""

import math
from beams_data import STEEL_E, STEEL_FY

PHI_C = 0.9  # compression resistance factor
PHI_B = 0.9  # bending resistance factor


def calculate_slenderness(height_ft, r_in, K=1.0):
    """
    Column slenderness ratio KL/r.

    Args:
        height_ft: Unbraced length in feet
        r_in:      Radius of gyration in inches (use ry for weak axis)
        K:         Effective length factor
    """
    return (K * height_ft * 12) / r_in


def calculate_critical_stress(KL_r, Fy=STEEL_FY, E=STEEL_E):
    """
    Critical buckling stress per AISC 360 Chapter E3.

    Returns:
        Fcr:           Critical stress (ksi)
        buckling_type: "inelastic" or "elastic"
    """
    Fe = math.pi**2 * E / KL_r**2
    limit = 4.71 * (E / Fy) ** 0.5

    if KL_r <= limit:
        Fcr = 0.658 ** (Fy / Fe) * Fy
        return Fcr, "inelastic"
    else:
        Fcr = 0.877 * Fe
        return Fcr, "elastic"


def check_axial_capacity(height_ft, Pu_kips, column, K=1.0,
                         Fy=STEEL_FY, E=STEEL_E):
    """
    Axial compression capacity check.

    Returns:
        passes:  Boolean
        ratio:   Pu / phi_Pn (<=1.0 passes)
        details: Dict with Pn, Fcr, KL_r, buckling_type
    """
    KL_r = calculate_slenderness(height_ft, column['ry'], K)
    Fcr, buckling_type = calculate_critical_stress(KL_r, Fy, E)

    Pn     = Fcr * column['A']
    phi_Pn = PHI_C * Pn
    ratio  = Pu_kips / phi_Pn

    details = {
        'Pn':            round(Pn, 2),
        'phi_Pn':        round(phi_Pn, 2),
        'Fcr':           round(Fcr, 2),
        'KL_r':          round(KL_r, 2),
        'buckling_type': buckling_type,
    }
    return ratio <= 1.0, ratio, details


def check_pm_interaction(Pu_kips, Mu_kipft, column, height_ft, K=1.0,
                         Fy=STEEL_FY, E=STEEL_E):
    """
    Combined axial + bending (P-M interaction) per AISC H1.

    Uses strong-axis moment capacity (Zx).
    """
    _, axial_ratio, axial_details = check_axial_capacity(
        height_ft, Pu_kips, column, K, Fy, E)

    phi_Pn = axial_details['phi_Pn']
    phi_Mn = PHI_B * column['Zx'] * Fy / 12  # kip-in -> kip-ft
    Pu_over_Pn = Pu_kips / phi_Pn
    Mu_over_Mn = Mu_kipft / phi_Mn if phi_Mn > 0 else 0.0

    if Pu_over_Pn >= 0.2:
        # AISC H1-1a
        ratio = Pu_over_Pn + (8.0 / 9.0) * Mu_over_Mn
        equation = "H1-1a"
    else:
        # AISC H1-1b
        ratio = Pu_over_Pn / 2.0 + Mu_over_Mn
        equation = "H1-1b"

    details = {
        'phi_Pn':      round(phi_Pn, 2),
        'phi_Mn':      round(phi_Mn, 2),
        'Pu_over_Pn':  round(Pu_over_Pn, 3),
        'Mu_over_Mn':  round(Mu_over_Mn, 3),
        'equation':    equation,
    }
    return ratio <= 1.0, ratio, details


def check_local_buckling_column(column, Fy=STEEL_FY, E=STEEL_E):
    """
    Flange and web compactness for members in compression (AISC Table B4.1a).

    Flange: lambda_f = (bf/2)/tf  <=  0.56*sqrt(E/Fy)
    Web:    lambda_w = (d-2*tf)/tw  <=  1.49*sqrt(E/Fy)
    """
    flange_limit = 0.56 * (E / Fy) ** 0.5
    web_limit    = 1.49 * (E / Fy) ** 0.5

    lambda_f = (column['bf'] / 2) / column['tf']
    lambda_w = (column['d'] - 2 * column['tf']) / column['tw']

    flange_ratio = lambda_f / flange_limit
    web_ratio    = lambda_w / web_limit

    passes = (flange_ratio <= 1.0) and (web_ratio <= 1.0)
    return passes, flange_ratio, web_ratio


def check_column_design(height_ft, Pu_kips, Mu_kipft, column, K=1.0,
                        Fy=STEEL_FY):
    """
    Master column check -- runs all checks.

    Returns:
        passes:       Boolean (all checks pass)
        weight:       lb/ft
        worst_name:   Controlling check name
        details:      Dict with all ratios
    """
    axial_ok, axial_ratio, axial_det = check_axial_capacity(
        height_ft, Pu_kips, column, K, Fy)

    pm_ok, pm_ratio, pm_det = check_pm_interaction(
        Pu_kips, Mu_kipft, column, height_ft, K, Fy)

    lb_ok, flange_ratio, web_ratio = check_local_buckling_column(
        column, Fy)

    passes = all([axial_ok, pm_ok, lb_ok])

    details = {
        'axial_ratio':   round(axial_ratio, 3),
        'pm_ratio':      round(pm_ratio, 3),
        'flange_ratio':  round(flange_ratio, 3),
        'web_ratio':     round(web_ratio, 3),
        'KL_r':          axial_det['KL_r'],
        'Fcr':           axial_det['Fcr'],
        'buckling_type': axial_det['buckling_type'],
        'phi_Pn':        pm_det['phi_Pn'],
        'phi_Mn':        pm_det['phi_Mn'],
        'pm_equation':   pm_det['equation'],
    }

    ratios = {
        'axial_ratio':  axial_ratio,
        'pm_ratio':     pm_ratio,
        'flange_ratio': flange_ratio,
        'web_ratio':    web_ratio,
    }
    worst_key  = max(ratios, key=ratios.get)
    worst_name = {
        'axial_ratio':  'axial capacity',
        'pm_ratio':     'P-M interaction',
        'flange_ratio': 'flange local buckling',
        'web_ratio':    'web local buckling',
    }[worst_key]

    return passes, column['weight'], worst_name, details


def find_lightest_passing_w14(height_ft, Pu_kips, Mu_kipft, K=1.0, Fy=STEEL_FY):
    """
    Lightest W14 in COLUMN_SECTIONS that passes all column checks.

    Returns:
        dict with keys name, max_ratio, details; or None if no W14 passes.
    """
    from beams_data import COLUMN_SECTIONS

    w14 = COLUMN_SECTIONS[
        COLUMN_SECTIONS['name'].str.startswith('W14')
    ].reset_index(drop=True)

    for _, row in w14.iterrows():
        col = row.to_dict()
        passes, weight, worst, details = check_column_design(
            height_ft, Pu_kips, Mu_kipft, col, K, Fy)
        if passes:
            ratios = [
                float(details['axial_ratio']),
                float(details['pm_ratio']),
                float(details['flange_ratio']),
                float(details['web_ratio']),
            ]
            max_r = max(ratios)
            return {'name': row['name'], 'max_ratio': max_r, 'details': details}
    return None


def explain_column_result(column_name, passes, details, alternatives=None):
    """Plain English explanation for column check results."""
    check_names = {
        'axial_ratio':  'axial capacity',
        'pm_ratio':     'P-M interaction',
        'flange_ratio': 'flange local buckling',
        'web_ratio':    'web local buckling',
    }

    if not passes:
        failed = {k: float(details[k]) for k in check_names
                  if k in details and float(details[k]) > 1.0}
        if failed:
            worst_key  = max(failed, key=failed.get)
            worst_name = check_names[worst_key]
            explanation = (f"{column_name} fails {worst_name} "
                          f"(utilization {failed[worst_key]:.2f} -- "
                          f"must be below 1.0).")
        else:
            explanation = f"{column_name} fails design checks."
    else:
        numeric = {k: float(details[k]) for k in check_names if k in details}
        controlling_key  = max(numeric, key=numeric.get)
        controlling_name = check_names[controlling_key]
        controlling_val  = numeric[controlling_key]
        explanation = (f"{column_name} passes all checks. "
                      f"Controlling: {controlling_name} "
                      f"at {controlling_val:.2f} utilization "
                      f"(KL/r = {details['KL_r']:.0f}, "
                      f"{details['buckling_type']} buckling, "
                      f"{details['pm_equation']}).")

    if alternatives:
        explanation += "\n\nAlternatives considered:"
        for alt_name, alt_passes, alt_weight in alternatives:
            status = "passes" if alt_passes else "fails"
            explanation += f"\n  {alt_name} ({alt_weight:.0f} lb/ft) -- {status}"

    return explanation


if __name__ == "__main__":
    from beams_data import BEAMS_DF

    print("Column Physics Engine Test (AISC 360)")
    print("=" * 60)

    tests = [
        ("Test 1: Pure axial",      "W12x65",  14, 300,   0, 1.0),
        ("Test 2: Axial + moment",  "W14x90",  20, 200, 150, 1.0),
        ("Test 3: Slender column",  "W14x90",  30, 150,  50, 1.0),
        ("Test 4: Short stocky",    "W12x65",  10, 400,  20, 1.0),
    ]

    for label, col_name, ht, pu, mu, K in tests:
        row = BEAMS_DF[BEAMS_DF['name'] == col_name]
        if row.empty:
            print(f"\n{label}: {col_name} not found")
            continue
        col = row.iloc[0].to_dict()
        passes, weight, worst, details = check_column_design(
            ht, pu, mu, col, K)
        print(f"\n{label}: {col_name} | {ht}ft | Pu={pu}k Mu={mu}k-ft | K={K}")
        print(f"  Passes: {passes}  Weight: {weight:.1f} lb/ft")
        print(f"  Controlling: {worst}")
        for k, v in details.items():
            print(f"  {k}: {v}")
        print(f"  Explanation: {explain_column_result(col_name, passes, details)}")
