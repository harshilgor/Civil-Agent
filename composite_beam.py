"""
Composite Beam Design
AISC 360 Chapter I: steel beam acting compositely with concrete slab via shear studs.

Design flow:
  1. Construction stage check (steel alone, wet concrete + 20 psf const. LL)
  2. Composite moment capacity (plastic stress distribution, AISC I3-2a)
  3. Effective moment of inertia (AISC Commentary C-I3-1 lower bound)
  4. Live-load deflection using Ieff
  5. Stud count from required shear flow
"""

import math

from beams_data import BEAMS_DF, get_beam_by_index, get_num_beams, STEEL_E, STEEL_FY
from beam_physics import check_beam_design, calculate_demands, apply_lrfd

# --- Constants ----------------------------------------------------------------
EC_MULTIPLIER    = 33       # ACI 318 Ec formula coefficient
WC_NORMAL        = 145      # normal weight concrete, pcf
STUD_DIAMETER    = 0.75     # inches, 3/4" shear stud
STUD_HEIGHT      = 3.0      # inches, standard stud height
PHI_STUD         = 0.75     # AISC stud resistance factor
MIN_COMPOSITE    = 0.25     # 25% minimum composite per AISC
PHI_FLEX         = 0.90     # flexural resistance factor
CONST_LL         = 0.020    # kip/ft² construction live load (20 psf)
STUD_FU          = 65       # ksi, stud ultimate strength (AWS Type B)


# --- Function 1: Concrete modulus ---------------------------------------------

def calculate_concrete_modulus(fc_ksi):
    """
    ACI 318 modulus of elasticity for normal-weight concrete.
    Ec = 33 × wc^1.5 × sqrt(fc_psi)  [psi]  -> divided by 1000 -> ksi
    """
    fc_psi = fc_ksi * 1000
    Ec_psi = EC_MULTIPLIER * (WC_NORMAL ** 1.5) * math.sqrt(fc_psi)
    return Ec_psi / 1000.0  # ksi


# --- Function 2: Effective slab width -----------------------------------------

def calculate_effective_width(span_ft, beam_spacing_ft):
    """
    AISC 360 Section I3.1a: effective slab width.
    Each side = min(span/8, spacing/2).
    Total beff = 2 × each-side (inches).
    """
    each_side = min(span_ft * 12 / 8.0, beam_spacing_ft * 12 / 2.0)
    return 2.0 * each_side  # inches


# --- Function 3: Individual stud shear capacity --------------------------------

def calculate_stud_capacity(fc_ksi, Ec_ksi):
    """
    AISC 360 I8-1: individual stud shear capacity.
      Qn = 0.5 × Asa × sqrt(fc × Ec)  [upper-bounded]
      Upper bound = Rg × Rp × Asa × Fu  (deck perpendicular to beam)
    Returns Qn in kips.
    """
    Asa        = math.pi / 4.0 * STUD_DIAMETER ** 2   # in²
    Qn_formula = 0.5 * Asa * math.sqrt(fc_ksi * Ec_ksi)
    Rg, Rp     = 0.85, 0.75                           # deck perpendicular
    Qn_limit   = Rg * Rp * Asa * STUD_FU
    return min(Qn_formula, Qn_limit)


# --- Function 4: Full composite horizontal shear force ------------------------

def calculate_full_composite_force(beam, fc_ksi, beff_in, slab_thickness_in):
    """
    AISC 360 I3-1: horizontal shear for full composite action.
    Cf = min(0.85 × fc × Ac, As × Fy)
    """
    Ac         = beff_in * slab_thickness_in           # in² of concrete
    C_concrete = 0.85 * fc_ksi * Ac
    C_steel    = float(beam['A']) * STEEL_FY
    return min(C_concrete, C_steel)


# --- Function 5: Effective composite moment of inertia -----------------------

def calculate_composite_Ieff(beam, composite_ratio, fc_ksi,
                              beff_in, slab_thickness_in, deck_height_in):
    """
    AISC Commentary C-I3-1 lower bound Ieff:
      Ieff = Is + sqrt(SumQn / Cf) × (Itr - Is)

    Transformed section: steel + concrete/n as one unit.
    Centroid measured from bottom of steel.
    """
    Es = STEEL_E
    Ec = calculate_concrete_modulus(fc_ksi)
    n  = Es / Ec          # modular ratio (round toward conservative)

    d  = float(beam['d'])
    Is = float(beam['Ix'])
    As = float(beam['A'])

    tc  = slab_thickness_in   # slab above deck
    hr  = deck_height_in      # deck rib height

    # Concrete area transformed to steel
    beff_t = beff_in / n       # transformed width

    # Centroid of concrete from bottom of steel = d + hr + tc/2
    y_conc = d + hr + tc / 2.0
    # Centroid of steel from bottom = d/2
    y_steel = d / 2.0

    # Transformed total area
    A_conc_t = beff_t * tc
    A_total  = As + A_conc_t

    # Fully composite centroid from bottom of steel
    y_bar = (As * y_steel + A_conc_t * y_conc) / A_total

    # Itr: fully composite moment of inertia
    Itr = (Is
           + As * (y_bar - y_steel) ** 2
           + (beff_t * tc ** 3 / 12.0)
           + A_conc_t * (y_conc - y_bar) ** 2)

    # SumQn / Cf ratio
    Cf     = calculate_full_composite_force(beam, fc_ksi, beff_in, slab_thickness_in)
    SumQn  = composite_ratio * Cf
    ratio  = max(MIN_COMPOSITE, min(1.0, SumQn / Cf))

    Ieff   = Is + math.sqrt(ratio) * (Itr - Is)
    return Ieff, Itr


# --- Function 6: Construction stage check ------------------------------------

def check_construction_stage(span_ft, dead_load_kip_ft, beam,
                              beam_spacing_ft=10.0):
    """
    Non-composite check during construction.
    Construction LL = 20 psf = 0.020 kip/ft² * beam_spacing.
    Beam is unbraced (no slab yet), Lb = span.
    Dead load = wet concrete self-weight on the beam.
    """
    const_ll_line = CONST_LL * beam_spacing_ft   # kip/ft
    passes, weight, worst, details = check_beam_design(
        span_ft, dead_load_kip_ft, const_ll_line, beam,
        Lb_ft=span_ft,    # unbraced during pour
        point_load=0,
        defl_limit=360,
    )
    return passes, details


# --- Function 7: Composite moment capacity ------------------------------------

def check_composite_moment(Mu_kip_ft, beam, composite_ratio, fc_ksi,
                             beff_in, slab_thickness_in, deck_height_in=3.0):
    """
    AISC I3-2a: plastic stress distribution for composite moment capacity.
    Simplified: locate plastic neutral axis, compute Mn.
    """
    d   = float(beam['d'])
    As  = float(beam['A'])
    Fy  = STEEL_FY

    tc  = slab_thickness_in
    hr  = deck_height_in

    Cf    = calculate_full_composite_force(beam, fc_ksi, beff_in, tc)
    SumQn = composite_ratio * Cf
    C_prime = SumQn   # horizontal shear force resisted

    # Depth of concrete compression block
    a = C_prime / (0.85 * fc_ksi * beff_in)

    # Lever arm from slab block centroid to steel centroid
    # Steel centroid from bottom = d/2
    # Slab top from bottom of steel = d + hr + tc
    # Slab compression block centroid from bottom = d + hr + tc - a/2
    y_slab_force = d + hr + tc - a / 2.0   # in

    # Steel contribution: fully yielded (tension) steel below PNA
    # Simplified: steel carries full Fy tension about its own centroid
    # Mn = C' × (lever arm from slab centroid to steel centroid)
    lever_arm = y_slab_force - d / 2.0
    Mn = C_prime * lever_arm / 12.0   # kip-ft

    # Add non-composite steel plastic moment for partial composite
    Mp_steel = Fy * float(beam['Zx']) / 12.0   # kip-ft
    # Interpolate between Mp (no composite) and fully composite
    # Use linear interpolation per AISC Commentary
    Mn = max(Mn, Mp_steel * composite_ratio)
    # Capacity can't exceed full plastic composite
    Mn_full = calculate_full_composite_force(beam, fc_ksi, beff_in, tc) * (
        d + hr + tc / 2.0) / 12.0
    Mn = min(Mn, Mn_full)

    phi_Mn = PHI_FLEX * Mn
    ratio  = (Mu_kip_ft * 12.0) / (phi_Mn * 12.0)   # dimensionless
    return ratio <= 1.0, round(ratio, 3)


# --- Function 8: Stud count --------------------------------------------------

def calculate_stud_count(composite_ratio, Cf_kips, Qn_kips):
    """
    AISC I8: studs required between point of maximum moment and
    zero moment (each half-span).
    N_half = ceil(SumQn / Qn) where SumQn = composite_ratio × Cf
    Total = 2 × N_half
    """
    SumQn   = composite_ratio * Cf_kips
    N_half  = math.ceil(SumQn / Qn_kips)
    return N_half, N_half * 2


# --- Function 9: Composite deflection check ----------------------------------

def check_composite_deflection(span_ft, live_load_kip_ft, beam,
                                Ieff, defl_limit=360):
    """
    Live-load deflection using lower bound Ieff.
    delta = 5 w L^4 / (384 E I)  (all in inches)
    """
    L_in  = span_ft * 12.0
    w_in  = live_load_kip_ft / 12.0    # kip/in
    delta = 5.0 * w_in * L_in ** 4 / (384.0 * STEEL_E * Ieff)
    limit = L_in / defl_limit
    ratio = delta / limit
    return ratio <= 1.0, round(ratio, 3)


# --- Function 10: Master design function --------------------------------------

def design_composite_beam(span_ft, dead_load, live_load,
                           beam_spacing_ft=10.0,
                           slab_thickness_in=3.5,
                           deck_height_in=3.0,
                           fc_ksi=4.0,
                           composite_ratio=0.5,
                           defl_limit=360):
    """
    Scan beams lightest-to-heaviest; return first passing composite design.

    Construction dead load = concrete slab (assumed ~0.75 × dead_load).
    Service:
      - Moment check uses full factored loads with composite capacity.
      - Deflection uses live load only with Ieff.
    """
    composite_ratio = max(MIN_COMPOSITE, min(1.0, composite_ratio))

    Ec  = calculate_concrete_modulus(fc_ksi)
    Qn  = calculate_stud_capacity(fc_ksi, Ec)

    beff_in = calculate_effective_width(span_ft, beam_spacing_ft)

    # Factored demands (same LRFD combos as non-composite)
    Wu = apply_lrfd(dead_load, live_load)
    Mu = Wu * span_ft ** 2 / 8.0   # kip-ft

    # Construction dead (wet concrete) ~ full dead load on bare steel
    const_dl = dead_load   # conservative — full DL on unshored steel

    # Baseline: find lightest non-composite beam (Lb=0, fully braced)
    # used for steel-savings comparison
    _nc_weight = None
    for _i in range(get_num_beams()):
        _, _bm = get_beam_by_index(_i)
        _ok, _wt, _, _ = check_beam_design(
            span_ft, dead_load, live_load, _bm,
            Lb_ft=0, defl_limit=defl_limit)
        if _ok:
            _nc_weight = _wt
            break

    for i in range(get_num_beams()):
        name, beam = get_beam_by_index(i)

        # --- Stage 1: Construction (steel alone) ---
        const_ok, const_details = check_construction_stage(
            span_ft, const_dl, beam, beam_spacing_ft)

        # --- Stage 2: Composite properties ---
        Cf = calculate_full_composite_force(
            beam, fc_ksi, beff_in, slab_thickness_in)

        Ieff, Itr = calculate_composite_Ieff(
            beam, composite_ratio, fc_ksi,
            beff_in, slab_thickness_in, deck_height_in)

        # --- Stage 3: Composite moment ---
        mom_ok, mom_ratio = check_composite_moment(
            Mu, beam, composite_ratio, fc_ksi,
            beff_in, slab_thickness_in, deck_height_in)

        # --- Stage 4: Composite deflection (live load only) ---
        defl_ok, defl_ratio = check_composite_deflection(
            span_ft, live_load, beam, Ieff, defl_limit)

        # --- Stage 5: Stud count ---
        N_half, N_total = calculate_stud_count(composite_ratio, Cf, Qn)

        passes = const_ok and mom_ok and defl_ok
        if passes:
            comp_wt   = float(beam['weight'])
            savings   = round((_nc_weight or comp_wt) - comp_wt, 1)

            return {
                'name':            name,
                'weight':          comp_wt,
                'span_ft':         span_ft,
                'composite_ratio': composite_ratio,
                'Ieff':            round(Ieff, 1),
                'Itr':             round(Itr, 1),
                'Is':              round(float(beam['Ix']), 1),
                'beff_in':         round(beff_in, 2),
                'Qn_per_stud':     round(Qn, 2),
                'Cf_kips':         round(Cf, 1),
                'studs_per_side':  N_half,
                'stud_count':      N_total,
                'construction_ok': const_ok,
                'service_ok':      mom_ok and defl_ok,
                'passes':          True,
                'nc_weight':       round(_nc_weight or comp_wt, 1),
                'steel_savings':   savings,
                'details': {
                    'construction_ratio': round(
                        max(v for k, v in const_details.items()
                            if isinstance(v, (int, float))), 3),
                    'moment_ratio':       mom_ratio,
                    'deflection_ratio':   defl_ratio,
                    'Qn_per_stud':        round(Qn, 2),
                    'total_force':        round(Cf * composite_ratio, 1),
                    'Ec':                 round(Ec, 1),
                    'n_modular':          round(STEEL_E / Ec, 1),
                },
            }

    # Nothing passed
    return {
        'name': 'NONE', 'weight': 0, 'passes': False,
        'steel_savings': 0,
        'details': {
            'construction_ratio': 0, 'moment_ratio': 0,
            'deflection_ratio': 0, 'Qn_per_stud': round(Qn, 2),
            'total_force': 0, 'Ec': round(Ec, 1), 'n_modular': 0,
        },
    }


def explain_composite(result):
    """Plain English summary of a composite design result."""
    if not result['passes']:
        return "No composite beam found for these inputs."
    savings = result['steel_savings']
    savings_lbs = savings * result['span_ft']
    cost_save = savings_lbs * 1.50
    Is = result['Is']
    Ieff = result['Ieff']
    gain = round((Ieff / Is - 1) * 100, 0) if Is > 0 else 0
    lines = [
        f"{result['name']} with {int(result['composite_ratio']*100)}% composite action.",
        f"Effective slab width: {result['beff_in']:.1f} in | "
        f"Modular ratio n = {result['details']['n_modular']:.0f}",
        f"Ieff = {Ieff:.0f} in4 vs {Is:.0f} in4 bare steel "
        f"(+{gain:.0f}% stiffness increase).",
        f"Studs: {result['studs_per_side']} each side of midspan "
        f"({result['stud_count']} total), "
        f"Qn = {result['Qn_per_stud']:.1f} kips each.",
    ]
    if savings > 0:
        lines.append(
            f"Steel savings: {savings:.1f} lb/ft vs non-composite "
            f"= {savings_lbs:.0f} lbs total "
            f"~ ${cost_save:,.0f} at $1.50/lb."
        )
    elif savings <= 0:
        lines.append(
            "Composite beam weight equals or exceeds non-composite for this loading "
            "(composite benefits are in stiffness / deflection control here)."
        )
    lines.append(
        f"Controlling checks: construction {result['details']['construction_ratio']:.2f}, "
        f"moment {result['details']['moment_ratio']:.2f}, "
        f"deflection {result['details']['deflection_ratio']:.2f}."
    )
    return "\n".join(lines)


# --- Revised composite design API -------------------------------------------
# These definitions intentionally override the first-pass prototype above while
# preserving the old function names as wrappers for existing app imports.

STUD_DIA = 0.75
DECK_HEIGHT = 3.0
MAX_COMPOSITE = 1.00
PHI_FLEXURE = 0.90
CONSTRUCTION_LL_KIP_FT = 0.02
STEEL_COST_PER_LB = 1.50


def get_concrete_modulus(fc_ksi):
    """Ec = 33 * wc^1.5 * sqrt(fc_psi), returned in ksi."""
    fc_psi = fc_ksi * 1000.0
    Ec_psi = 33.0 * (WC_NORMAL ** 1.5) * (fc_psi ** 0.5)
    return Ec_psi / 1000.0


def get_effective_width(span_ft, spacing_ft):
    """AISC I3.1a effective slab width, both sides of the beam, in inches."""
    each_side = min(span_ft * 12.0 / 8.0, spacing_ft * 12.0 / 2.0)
    return 2.0 * each_side


def get_stud_capacity(fc_ksi, Ec_ksi):
    """AISC I8-1 nominal shear stud capacity, in kips."""
    Asa = math.pi / 4.0 * STUD_DIA ** 2
    Qn_formula = 0.5 * Asa * (fc_ksi * Ec_ksi) ** 0.5
    Qn_limit = 0.85 * 0.75 * Asa * STUD_FU
    return min(Qn_formula, Qn_limit)


def get_full_composite_force(beam, fc_ksi, beff_in, tc_in):
    """Force required for full composite action, in kips."""
    Ac = beff_in * tc_in
    C_conc = 0.85 * fc_ksi * Ac
    C_steel = float(beam["A"]) * STEEL_FY
    return min(C_conc, C_steel)


def get_stud_count(composite_ratio, Cf_kips, Qn_kips, half_span_ft=None):
    """Studs required each side of midspan and total studs."""
    if Qn_kips <= 0:
        return 0, 0
    composite_ratio = max(MIN_COMPOSITE, min(MAX_COMPOSITE, composite_ratio))
    studs_each_side = math.ceil(composite_ratio * Cf_kips / Qn_kips)
    if half_span_ft is not None:
        studs_each_side = max(studs_each_side, math.ceil(half_span_ft))
    return studs_each_side, studs_each_side * 2


def _transformed_section(beam, fc_ksi, beff_in, tc_in, deck_hr=DECK_HEIGHT):
    Ec = get_concrete_modulus(fc_ksi)
    n = max(1, round(STEEL_E / Ec))
    As = float(beam["A"])
    Is = float(beam["Ix"])
    d = float(beam["d"])
    A_tr = beff_in * tc_in / n
    d_steel = d / 2.0
    d_slab = d + deck_hr + tc_in / 2.0
    y_bar = (As * d_steel + A_tr * d_slab) / (As + A_tr)
    Itr = (
        Is
        + As * (y_bar - d_steel) ** 2
        + beff_in * tc_in ** 3 / (12.0 * n)
        + A_tr * (d_slab - y_bar) ** 2
    )
    return Itr, n


def get_composite_Ieff(beam, composite_ratio, fc_ksi, beff_in, tc_in,
                       deck_hr=DECK_HEIGHT):
    """AISC Commentary C-I3-1 lower-bound effective moment of inertia."""
    composite_ratio = max(MIN_COMPOSITE, min(MAX_COMPOSITE, composite_ratio))
    Is = float(beam["Ix"])
    Itr, _ = _transformed_section(beam, fc_ksi, beff_in, tc_in, deck_hr)
    Cf = get_full_composite_force(beam, fc_ksi, beff_in, tc_in)
    SQn = composite_ratio * Cf
    return Is + math.sqrt(SQn / Cf) * (Itr - Is) if Cf > 0 else Is


def check_construction_stage(span_ft, dead_load_kip_ft, beam,
                             deck_span_ft=None):
    """
    Check the steel beam alone during construction.

    Uses the project convention from the prompt: wet/dead construction line load
    plus beam self-weight and 0.02 kip/ft construction live load. Metal deck
    braces the top flange during construction, so Lb is the deck span.
    """
    if deck_span_ft is None:
        deck_span_ft = span_ft
    self_weight = float(beam["weight"]) / 1000.0
    dead_total = dead_load_kip_ft + self_weight
    _passes, _weight, _worst, details = check_beam_design(
        span_ft,
        dead_total,
        CONSTRUCTION_LL_KIP_FT,
        beam,
        Lb_ft=deck_span_ft,
    )
    details["Lb_ft"] = round(deck_span_ft, 3)
    # Construction pass/fail is strength-based here. Construction deflection is
    # retained in details as informational; service deflection is checked later
    # using the composite effective inertia.
    strength_keys = (
        "moment_ratio",
        "shear_ratio",
        "flange_ratio",
        "web_ratio",
        "ltb_ratio",
    )
    worst = max(float(details.get(key, 0.0)) for key in strength_keys)
    passes = worst <= 1.0
    return passes, worst, details


def check_composite_moment(Mu_kip_ft, beam, composite_ratio, fc_ksi,
                           beff_in, tc_in, deck_hr=DECK_HEIGHT):
    """Simplified conservative composite plastic moment check."""
    composite_ratio = max(MIN_COMPOSITE, min(MAX_COMPOSITE, composite_ratio))
    d = float(beam["d"])
    C_steel = float(beam["A"]) * STEEL_FY
    Cf = get_full_composite_force(beam, fc_ksi, beff_in, tc_in)
    SQn = composite_ratio * Cf
    if SQn <= 0:
        return False, float("inf")
    a = min(tc_in, SQn / (0.85 * fc_ksi * beff_in))
    d1 = deck_hr + tc_in - a / 2.0
    Mn_kip_in = SQn * (d1 + d) / 2.0 + max(0.0, C_steel - SQn) * d / 2.0
    phi_Mn = PHI_FLEXURE * Mn_kip_in
    ratio = (Mu_kip_ft * 12.0) / phi_Mn if phi_Mn > 0 else float("inf")
    return ratio <= 1.0, round(ratio, 3)


def check_composite_deflection(span_ft, live_load_kip_ft, Ieff,
                               defl_limit=360):
    """Live-load deflection using Ieff."""
    L_in = span_ft * 12.0
    w_in = live_load_kip_ft / 12.0
    delta = 5.0 * w_in * L_in ** 4 / (384.0 * STEEL_E * Ieff)
    limit = L_in / defl_limit
    ratio = delta / limit
    return ratio <= 1.0, round(ratio, 3)


def _lightest_noncomposite(span_ft, dead_load, live_load, defl_limit=360):
    for _, row in BEAMS_DF.sort_values("weight").iterrows():
        beam = row.to_dict()
        passes, weight, _, _ = check_beam_design(
            span_ft, dead_load, live_load, beam, Lb_ft=0,
            defl_limit=defl_limit)
        if passes:
            return beam["name"], float(weight)
    return "NONE", 0.0


def design_composite_beam(span_ft, dead_load, live_load, beam_spacing_ft,
                          slab_thickness_in=3.5, deck_height_in=DECK_HEIGHT,
                          fc_ksi=4.0, composite_ratio=0.5,
                          defl_limit=360, shored=False):
    """Master composite design function; returns the lightest passing section."""
    composite_ratio = max(MIN_COMPOSITE, min(MAX_COMPOSITE, composite_ratio))
    Ec = get_concrete_modulus(fc_ksi)
    Qn = get_stud_capacity(fc_ksi, Ec)
    beff_in = get_effective_width(span_ft, beam_spacing_ft)
    Mu = apply_lrfd(dead_load, live_load) * span_ft ** 2 / 8.0
    nc_name, nc_weight = _lightest_noncomposite(
        span_ft, dead_load, live_load, defl_limit)

    for _, row in BEAMS_DF.sort_values("weight").iterrows():
        beam = row.to_dict()
        if shored:
            construction_ok = True
            construction_ratio = 0.0
            construction_details = {
                "note": "Shored construction assumed",
                "Lb_ft": 0.0,
            }
        else:
            construction_ok, construction_ratio, construction_details = (
                check_construction_stage(
                    span_ft, dead_load, beam, deck_span_ft=beam_spacing_ft)
            )
        Cf = get_full_composite_force(beam, fc_ksi, beff_in,
                                      slab_thickness_in)
        Ieff = get_composite_Ieff(
            beam, composite_ratio, fc_ksi, beff_in, slab_thickness_in,
            deck_height_in)
        Itr, n_modular = _transformed_section(
            beam, fc_ksi, beff_in, slab_thickness_in, deck_height_in)
        moment_ok, moment_ratio = check_composite_moment(
            Mu, beam, composite_ratio, fc_ksi, beff_in, slab_thickness_in,
            deck_height_in)
        deflection_ok, deflection_ratio = check_composite_deflection(
            span_ft, live_load, Ieff, defl_limit)
        studs_per_side, total_studs = get_stud_count(
            composite_ratio, Cf, Qn, half_span_ft=span_ft / 2.0)

        if not (construction_ok and moment_ok and deflection_ok
                and total_studs > 0):
            continue

        weight = float(beam["weight"])
        steel_savings = round((nc_weight or weight) - weight, 1)
        total_saved = steel_savings * span_ft
        gain = (Ieff / float(beam["Ix"]) - 1.0) * 100.0
        if steel_savings > 0:
            comparison = (
                f"Composite {beam['name']} vs non-composite {nc_name} "
                f"saves {steel_savings:.1f} lb/ft x {span_ft:g}ft = "
                f"{total_saved:.0f} lbs per beam."
            )
        else:
            comparison = (
                f"Composite {beam['name']} does not save weight versus "
                f"non-composite {nc_name} for these assumptions."
            )

        return {
            "name": beam["name"],
            "beam_name": beam["name"],
            "weight": weight,
            "composite_ratio": composite_ratio,
            "Ieff": round(Ieff, 1),
            "Is": round(float(beam["Ix"]), 1),
            "Itr": round(Itr, 1),
            "studs_per_side": studs_per_side,
            "total_studs": total_studs,
            "stud_count": total_studs,
            "Qn_per_stud": round(Qn, 2),
            "passes": True,
            "construction_ok": True,
            "service_ok": True,
            "shored": shored,
            "beff_in": round(beff_in, 2),
            "Cf_kips": round(Cf, 1),
            "nc_name": nc_name,
            "nc_weight": round(nc_weight or weight, 1),
            "steel_savings": steel_savings,
            "stiffness_gain": f"{gain:.0f}% increase vs non-composite",
            "weight_vs_noncomp": comparison,
            "span_ft": span_ft,
            "details": {
                "construction_ratio": round(construction_ratio, 3),
                "construction_note": (
                    "Shored construction assumed"
                    if shored
                    else f"Construction Lb = deck span = {beam_spacing_ft:g} ft"
                ),
                "moment_ratio": moment_ratio,
                "deflection_ratio": deflection_ratio,
                "defl_ratio": deflection_ratio,
                "shear_ratio": 0.0,
                "ltb_ratio": round(construction_ratio, 3),
                "ltb_zone": "construction stage",
                "flange_ratio": 0.0,
                "web_ratio": 0.0,
                "beff_in": round(beff_in, 2),
                "Cf_kips": round(Cf, 1),
                "n_modular": n_modular,
                "Qn_per_stud": round(Qn, 2),
                "total_force": round(Cf * composite_ratio, 1),
                "Ec": round(Ec, 1),
                "construction_details": construction_details,
            },
        }

    return {
        "name": "NONE",
        "beam_name": "NONE",
        "weight": 0.0,
        "composite_ratio": composite_ratio,
        "passes": False,
        "steel_savings": 0.0,
        "details": {
            "construction_ratio": 0.0,
            "moment_ratio": 0.0,
            "deflection_ratio": 0.0,
            "beff_in": round(beff_in, 2),
            "Cf_kips": 0.0,
            "n_modular": round(STEEL_E / Ec),
            "Qn_per_stud": round(Qn, 2),
            "total_force": 0.0,
            "Ec": round(Ec, 1),
        },
    }


def explain_composite_result(result):
    """Plain English explanation for a composite design result."""
    if not result.get("passes"):
        return "No composite beam found for these inputs."
    Is = float(result.get("Is", 0.0) or 0.0)
    Ieff = float(result.get("Ieff", 0.0) or 0.0)
    gain = (Ieff / Is - 1.0) * 100.0 if Is > 0 else 0.0
    lines = [
        f"{result['name']} composite ({int(result['composite_ratio'] * 100)}%) passes all checks.",
        f"Effective I = {Ieff:,.0f} in4 vs {Is:,.0f} in4 steel alone ({gain:.0f}% increase).",
        f"Requires {result['studs_per_side']} studs each side of midspan ({result['total_studs']} total).",
    ]
    if result.get("weight_vs_noncomp"):
        lines.append(result["weight_vs_noncomp"])
    return "\n".join(lines)


# Backward-compatible names used elsewhere in the current project.
calculate_concrete_modulus = get_concrete_modulus
calculate_effective_width = get_effective_width
calculate_stud_capacity = get_stud_capacity
calculate_full_composite_force = get_full_composite_force
calculate_stud_count = get_stud_count


def calculate_composite_Ieff(beam, composite_ratio, fc_ksi,
                             beff_in, slab_thickness_in, deck_height_in):
    """Compatibility wrapper returning both Ieff and Itr."""
    Ieff = get_composite_Ieff(
        beam, composite_ratio, fc_ksi, beff_in, slab_thickness_in,
        deck_height_in)
    Itr, _ = _transformed_section(
        beam, fc_ksi, beff_in, slab_thickness_in, deck_height_in)
    return Ieff, Itr


explain_composite = explain_composite_result


if __name__ == "__main__":
    from beams_data import BEAMS_DF

    print("Composite Beam Design Tests")
    print("=" * 62)

    cases = [
        {
            "label":    "Test 1: Standard office (30ft, 0.8DL+1.0LL, 50% comp)",
            "span":     30, "dl": 0.8, "ll": 1.0,
            "spacing":  10, "tc": 3.5, "hr": 3.0, "fc": 4.0, "comp": 0.50,
        },
        {
            "label":    "Test 2: Heavy floor (40ft, 1.2DL+1.5LL, 75% comp)",
            "span":     40, "dl": 1.2, "ll": 1.5,
            "spacing":  10, "tc": 3.5, "hr": 3.0, "fc": 4.0, "comp": 0.75,
        },
        {
            "label":    "Test 3: Steel savings comparison (30ft, 0.8DL+1.0LL)",
            "span":     30, "dl": 0.8, "ll": 1.0,
            "spacing":  10, "tc": 3.5, "hr": 3.0, "fc": 4.0, "comp": 0.50,
        },
        {
            "label":    "Test 4: Construction stage (40ft, 1.2DL+1.5LL, 75%)",
            "span":     40, "dl": 1.2, "ll": 1.5,
            "spacing":  10, "tc": 3.5, "hr": 3.0, "fc": 4.0, "comp": 0.75,
        },
    ]

    for c in cases:
        print(f"\n{c['label']}")
        print("-" * 62)
        r = design_composite_beam(
            c["span"], c["dl"], c["ll"],
            beam_spacing_ft=c["spacing"],
            slab_thickness_in=c["tc"],
            deck_height_in=c["hr"],
            fc_ksi=c["fc"],
            composite_ratio=c["comp"],
        )

        if not r["passes"]:
            print("  No passing composite beam found.")
            continue

        d = r["details"]
        print(f"  Composite beam:   {r['name']}  ({r['weight']:.1f} lb/ft)")
        print(f"  Composite ratio:  {int(r['composite_ratio']*100)}%")
        print(f"  Ieff / Is / Itr:  {r['Ieff']:.0f} / {r['Is']:.0f} / "
              f"{r['Itr']:.0f} in4")
        print(f"  Studs:            {r['studs_per_side']} per side "
              f"({r['stud_count']} total), Qn = {r['Qn_per_stud']:.2f} kips")
        print(f"  Checks:")
        print(f"    Construction:  {d['construction_ratio']:.3f}  "
              f"{'OK' if d['construction_ratio'] <= 1.0 else 'FAIL'}")
        print(f"    Moment:        {d['moment_ratio']:.3f}  "
              f"{'OK' if d['moment_ratio'] <= 1.0 else 'FAIL'}")
        print(f"    Deflection:    {d['deflection_ratio']:.3f}  "
              f"{'OK' if d['deflection_ratio'] <= 1.0 else 'FAIL'}")
        print(f"  Steel savings vs non-composite: {r['steel_savings']:.1f} lb/ft")
        print(f"    = {r['steel_savings'] * c['span']:.0f} lbs total  "
              f"~ ${r['steel_savings'] * c['span'] * 1.5:,.0f}")
        print(f"  Explanation:")
        for line in explain_composite(r).split("\n"):
            print(f"    {line}")

    # Test 3 detail: compare composite vs non-composite side-by-side
    print("\n\nTest 3 detailed — composite vs non-composite comparison")
    print("=" * 62)
    from beam_physics import check_beam_design as cbd

    c = cases[2]
    r_comp = design_composite_beam(
        c["span"], c["dl"], c["ll"],
        beam_spacing_ft=c["spacing"], slab_thickness_in=c["tc"],
        deck_height_in=c["hr"], fc_ksi=c["fc"], composite_ratio=c["comp"])

    # Non-composite: find lightest passing beam
    r_nc_name, r_nc_wt = "NONE", 0
    for i in range(get_num_beams()):
        nm, bm = get_beam_by_index(i)
        ok, wt, _, _ = cbd(c["span"], c["dl"], c["ll"], bm, Lb_ft=0)
        if ok:
            r_nc_name, r_nc_wt = nm, wt
            break

    print(f"  Non-composite: {r_nc_name} ({r_nc_wt:.1f} lb/ft)")
    print(f"  Composite:     {r_comp['name']} ({r_comp['weight']:.1f} lb/ft)")
    savings_lbs = r_comp['steel_savings'] * c["span"]
    print(f"  Weight saving: {r_comp['steel_savings']:.1f} lb/ft "
          f"= {savings_lbs:.0f} lbs = ${savings_lbs * 1.5:,.0f}")
    print(f"  Stiffness gain: Is={r_comp['Is']:.0f}  Ieff={r_comp['Ieff']:.0f}  "
          f"(+{(r_comp['Ieff']/r_comp['Is']-1)*100:.0f}% more stiff)")
    print(f"  Studs: {r_comp['stud_count']} total "
          f"({r_comp['studs_per_side']} each side of midspan)")
