"""
Floor System Design
Sequential: beams -> girder -> column.
Beams sized first, girder from beam reactions, column from tributary area.
"""

from beams_data import (BEAMS_DF, COLUMN_SECTIONS,
                        get_num_beams, get_beam_by_index,
                        get_num_columns, get_column_by_index)
from beam_physics import check_beam_design, calculate_demands, apply_lrfd
from column_physics import check_column_design
from foundation_design import design_base_plate, design_spread_footing
from slab_design import design_composite_slab

STEEL_E  = 29000
STEEL_FY = 50


class FloorSystem:

    def __init__(self, bay_length_ft, bay_width_ft,
                 dead_load_psf, live_load_psf,
                 beam_spacing_ft=10, num_floors=1,
                 floor_height_ft=14, defl_limit=360,
                 composite_beams=True,
                 slab_thickness_in=3.5,
                 deck_height_in=3.0,
                 fc_ksi=4.0,
                 composite_ratio=0.5):
        self.bay_length_ft      = bay_length_ft
        self.bay_width_ft       = bay_width_ft
        self.dead_load_psf      = dead_load_psf
        self.live_load_psf      = live_load_psf
        self.beam_spacing_ft    = beam_spacing_ft
        self.num_floors         = num_floors
        self.floor_height_ft    = floor_height_ft
        self.defl_limit         = defl_limit
        self.composite_beams    = composite_beams
        self.slab_thickness_in  = slab_thickness_in
        self.deck_height_in     = deck_height_in
        self.fc_ksi             = fc_ksi
        self.composite_ratio    = composite_ratio

        self.results = {
            'beams': None, 'girder': None, 'column': None,
            'slab': None, 'footing': None, 'base_plate': None,
            'total_weight': 0, 'passes': False,
        }

    def design_beams(self):
        """Design floor beams spanning bay_width, tributary width = beam_spacing."""
        DL_line = self.dead_load_psf * self.beam_spacing_ft / 1000  # kip/ft
        LL_line = self.live_load_psf * self.beam_spacing_ft / 1000
        span    = self.bay_width_ft
        n_beams = max(1, int(self.bay_length_ft / self.beam_spacing_ft) - 1)

        if self.composite_beams:
            from composite_beam import design_composite_beam
            r = design_composite_beam(
                span, DL_line, LL_line,
                beam_spacing_ft=self.beam_spacing_ft,
                slab_thickness_in=self.slab_thickness_in,
                deck_height_in=self.deck_height_in,
                fc_ksi=self.fc_ksi,
                composite_ratio=self.composite_ratio,
                defl_limit=self.defl_limit,
                shored=False,
            )
            if r['passes']:
                _, Vu, _ = calculate_demands(span, DL_line, LL_line)
                self.results['beams'] = {
                    'name': r['name'], 'weight': r['weight'],
                    'span': span, 'spacing': self.beam_spacing_ft,
                    'DL_line': round(DL_line, 4),
                    'LL_line': round(LL_line, 4),
                    'num_beams': n_beams,
                    'reaction': round(Vu, 1),
                    'details': r['details'], 'passes': True,
                    'composite': True,
                    'studs_per_side': r.get('studs_per_side', 0),
                    'total_studs': r.get('total_studs', r.get('stud_count', 0)),
                    'stud_count': r.get('stud_count', 0),
                    'Ieff': r.get('Ieff', 0),
                    'steel_savings': r.get('steel_savings', 0),
                    'noncomp_name': r.get('nc_name', 'NONE'),
                    'noncomp_weight': r.get('nc_weight', r['weight']),
                }
                return self.results['beams']
            # fall through to non-composite scan if composite fails
            self.composite_beams = False

        Lb_ft = 0  # slab provides continuous lateral bracing

        for i in range(get_num_beams()):
            name, beam = get_beam_by_index(i)
            passes, weight, _, details = check_beam_design(
                span, DL_line, LL_line, beam,
                Lb_ft=Lb_ft, point_load=0, defl_limit=self.defl_limit)
            if passes:
                _, Vu, _ = calculate_demands(span, DL_line, LL_line)
                self.results['beams'] = {
                    'name': name, 'weight': weight, 'span': span,
                    'spacing': self.beam_spacing_ft,
                    'DL_line': round(DL_line, 4),
                    'LL_line': round(LL_line, 4),
                    'num_beams': n_beams,
                    'reaction': round(Vu, 1),
                    'details': details, 'passes': True,
                    'composite': False,
                    'noncomp_name': name,
                    'noncomp_weight': weight,
                    'steel_savings': 0.0,
                }
                return self.results['beams']

        self.results['beams'] = {
            'name': 'NONE', 'weight': 0, 'span': span,
            'spacing': self.beam_spacing_ft,
            'DL_line': round(DL_line, 4),
            'LL_line': round(LL_line, 4),
            'num_beams': n_beams, 'reaction': 0,
            'details': {}, 'passes': False, 'composite': False,
        }
        return self.results['beams']

    def design_girder(self):
        """Design girder spanning bay_length; point loads from beam reactions."""
        if self.results['beams'] is None:
            self.design_beams()
        b       = self.results['beams']
        n_beams = b['num_beams']

        R_DL = b['DL_line'] * self.bay_width_ft / 2   # unfactored beam end reaction
        R_LL = b['LL_line'] * self.bay_width_ft / 2
        factored_pt = round(apply_lrfd(R_DL, R_LL), 1)

        DL_girder = n_beams * R_DL / self.bay_length_ft  # equiv uniform (conservative)
        LL_girder = n_beams * R_LL / self.bay_length_ft

        span  = self.bay_length_ft
        Lb_ft = self.beam_spacing_ft  # beams brace the girder
        noncomp_girder = None

        for i in range(get_num_beams()):
            name, girder = get_beam_by_index(i)
            passes, weight, _, details = check_beam_design(
                span, DL_girder, LL_girder, girder,
                Lb_ft=Lb_ft, point_load=0, defl_limit=self.defl_limit)
            if passes:
                noncomp_girder = {
                    'name': name,
                    'weight': weight,
                    'details': details,
                }
                break

        if self.composite_beams:
            from composite_beam import design_composite_beam
            r = design_composite_beam(
                span, DL_girder, LL_girder,
                beam_spacing_ft=self.beam_spacing_ft,
                slab_thickness_in=self.slab_thickness_in,
                deck_height_in=self.deck_height_in,
                fc_ksi=self.fc_ksi,
                composite_ratio=self.composite_ratio,
                defl_limit=self.defl_limit,
                shored=False,
            )
            if r['passes']:
                _, Vu, _ = calculate_demands(span, DL_girder, LL_girder)
                baseline_weight = (
                    noncomp_girder['weight'] if noncomp_girder
                    else r.get('nc_weight', r['weight'])
                )
                baseline_name = (
                    noncomp_girder['name'] if noncomp_girder
                    else r.get('nc_name', 'NONE')
                )
                self.results['girder'] = {
                    'name': r['name'], 'weight': r['weight'],
                    'span': span,
                    'n_point_loads': n_beams,
                    'point_load_kips': factored_pt,
                    'equiv_DL': round(DL_girder, 4),
                    'equiv_LL': round(LL_girder, 4),
                    'reaction': round(Vu, 1),
                    'Lb_ft': Lb_ft,
                    'details': r['details'], 'passes': True,
                    'composite': True,
                    'studs_per_side': r.get('studs_per_side', 0),
                    'total_studs': r.get('total_studs', r.get('stud_count', 0)),
                    'stud_count': r.get('stud_count', 0),
                    'Ieff': r.get('Ieff', 0),
                    'noncomp_name': baseline_name,
                    'noncomp_weight': round(baseline_weight, 2),
                    'steel_savings': round(baseline_weight - r['weight'], 1),
                }
                return self.results['girder']

        if noncomp_girder is not None:
            _, Vu, _ = calculate_demands(span, DL_girder, LL_girder)
            self.results['girder'] = {
                'name': noncomp_girder['name'],
                'weight': noncomp_girder['weight'],
                'span': span,
                'n_point_loads': n_beams,
                'point_load_kips': factored_pt,
                'equiv_DL': round(DL_girder, 4),
                'equiv_LL': round(LL_girder, 4),
                'reaction': round(Vu, 1),
                'Lb_ft': Lb_ft,
                'details': noncomp_girder['details'],
                'passes': True,
                'composite': False,
                'noncomp_name': noncomp_girder['name'],
                'noncomp_weight': round(noncomp_girder['weight'], 2),
                'steel_savings': 0.0,
            }
            return self.results['girder']

        for i in range(get_num_beams()):
            name, girder = get_beam_by_index(i)
            passes, weight, _, details = check_beam_design(
                span, DL_girder, LL_girder, girder,
                Lb_ft=Lb_ft, point_load=0, defl_limit=self.defl_limit)
            if passes:
                _, Vu, _ = calculate_demands(span, DL_girder, LL_girder)
                self.results['girder'] = {
                    'name': name, 'weight': weight, 'span': span,
                    'n_point_loads': n_beams,
                    'point_load_kips': factored_pt,
                    'equiv_DL': round(DL_girder, 4),
                    'equiv_LL': round(LL_girder, 4),
                    'reaction': round(Vu, 1),
                    'Lb_ft': Lb_ft,
                    'details': details, 'passes': True,
                    'composite': False,
                }
                return self.results['girder']

        self.results['girder'] = {
            'name': 'NONE', 'weight': 0, 'span': span,
            'n_point_loads': n_beams, 'point_load_kips': factored_pt,
            'equiv_DL': 0, 'equiv_LL': 0, 'reaction': 0,
            'Lb_ft': Lb_ft, 'details': {}, 'passes': False,
        }
        return self.results['girder']

    def design_column(self):
        """Design column using tributary area for axial load."""
        if self.results['girder'] is None:
            self.design_girder()

        Wu_psf   = 1.2 * self.dead_load_psf + 1.6 * self.live_load_psf
        Pu_floor = Wu_psf * self.bay_length_ft * self.bay_width_ft / 1000
        Pu_total = Pu_floor * self.num_floors
        Mu       = 0.01 * Pu_total * self.floor_height_ft

        height = self.floor_height_ft
        K      = 1.0

        for i in range(get_num_columns()):
            name, col = get_column_by_index(i)
            passes, weight, _, details = check_column_design(
                height, Pu_total, Mu, col, K)
            if passes:
                self.results['column'] = {
                    'name': name, 'weight': weight, 'height': height,
                    'Pu': round(Pu_total, 1), 'Mu': round(Mu, 1),
                    'K': K, 'details': details, 'passes': True,
                }
                return self.results['column']

        self.results['column'] = {
            'name': 'NONE', 'weight': 0, 'height': height,
            'Pu': round(Pu_total, 1), 'Mu': round(Mu, 1),
            'K': K, 'details': {}, 'passes': False,
        }
        return self.results['column']

    def design_slab(self):
        slab = design_composite_slab(
            beam_spacing_ft=self.beam_spacing_ft,
            total_slab_thickness_in=self.deck_height_in + self.slab_thickness_in,
            dead_load_psf=self.dead_load_psf * 0.6,
            live_load_psf=self.live_load_psf,
            deck_type="3B" if self.deck_height_in >= 3.0 else "2B",
            fc_ksi=self.fc_ksi,
        )
        self.results["slab"] = slab
        return slab

    def design_foundation(self):
        if self.results["column"] is None:
            self.design_column()
        col = self.results["column"]
        footing = design_spread_footing(
            Pu_kips=col["Pu"],
            Mu_kip_ft=col["Mu"],
            soil_bearing_ksf=2.0,
            fc_ksi=self.fc_ksi,
            column_width_in=14.0,
            depth_ft=4.0,
        )
        base_plate = design_base_plate(
            Pu_kips=col["Pu"],
            Mu_kip_ft=col["Mu"],
            column={"bf": 14.0, "d": 14.0, "tf": 0.75, "tw": 0.5, **col},
        )
        self.results["footing"] = footing
        self.results["base_plate"] = base_plate
        return footing, base_plate

    def design_all(self):
        """Run beam -> girder -> column in sequence; compute total weight."""
        self.design_beams()
        self.design_girder()
        self.design_column()
        self.design_slab()
        self.design_foundation()

        b = self.results['beams']
        g = self.results['girder']
        c = self.results['column']

        beam_total   = b['weight'] * b['span'] * b['num_beams']
        girder_total = g['weight'] * g['span']
        column_total = c['weight'] * c['height']
        beam_noncomp_total = (
            b.get('noncomp_weight', b['weight']) * b['span'] * b['num_beams']
        )
        girder_noncomp_total = (
            g.get('noncomp_weight', g['weight']) * g['span']
        )
        noncomp_total = beam_noncomp_total + girder_noncomp_total + column_total
        total_savings = noncomp_total - (beam_total + girder_total + column_total)

        self.results['total_weight'] = round(
            beam_total + girder_total + column_total, 0)
        self.results['noncomp_total_weight'] = round(noncomp_total, 0)
        self.results['composite_savings'] = {
            'beam_composite_lbs': round(beam_total, 0),
            'beam_noncomp_lbs': round(beam_noncomp_total, 0),
            'beam_savings_lbs': round(beam_noncomp_total - beam_total, 0),
            'girder_composite_lbs': round(girder_total, 0),
            'girder_noncomp_lbs': round(girder_noncomp_total, 0),
            'girder_savings_lbs': round(girder_noncomp_total - girder_total, 0),
            'total_savings_lbs': round(total_savings, 0),
            'cost_savings': round(total_savings * 1.50, 0),
        }
        self.results['passes'] = all([b['passes'], g['passes'], c['passes']])
        self.results['full_system_passes'] = all([
            b['passes'],
            g['passes'],
            c['passes'],
            self.results["slab"]["passes"],
            self.results["footing"]["passes"],
            self.results["base_plate"]["passes"],
        ])

        return self.results

    def get_summary(self):
        b = self.results['beams']
        g = self.results['girder']
        c = self.results['column']
        total = self.dead_load_psf + self.live_load_psf
        status = "PASS" if self.results['passes'] else "FAIL"

        beam_wt   = b['weight'] * b['span'] * b['num_beams']
        girder_wt = g['weight'] * g['span']
        column_wt = c['weight'] * c['height']
        savings = self.results.get('composite_savings', {})
        slab = self.results.get("slab") or {}
        footing = self.results.get("footing") or {}
        base_plate = self.results.get("base_plate") or {}

        lines = [
            "FLOOR SYSTEM DESIGN SUMMARY",
            "=" * 44,
            f"Bay: {self.bay_length_ft:g}ft x {self.bay_width_ft:g}ft",
            f"Loads: {self.dead_load_psf:g} psf DL + "
            f"{self.live_load_psf:g} psf LL = {total:g} psf total",
            "",
            f"BEAMS ({self.bay_width_ft:g}ft span, "
            f"{self.beam_spacing_ft:g}ft o.c.)",
            f"  Section:  {b['name']}  ({b['weight']:.1f} lb/ft)"
            + (" [COMPOSITE]" if b.get('composite') else ""),
            f"  Count:    {b['num_beams']}",
            f"  Line DL:  {b['DL_line']:.3f} kip/ft  |  "
            f"Line LL: {b['LL_line']:.3f} kip/ft",
            f"  Reaction: {b['reaction']:.1f} kips each end (factored)",
        ] + ([
            f"  Studs:    {b['stud_count']} total ({b['stud_count']//2} per side)  |  "
            f"Ieff: {b['Ieff']:.0f} in4",
        ] if b.get('composite') else []) + [
            f"  Status:   {'PASS' if b['passes'] else 'FAIL'}",
            "",
            f"GIRDER ({self.bay_length_ft:g}ft span, "
            f"Lb = {g['Lb_ft']:g}ft)",
            f"  Section:    {g['name']}  ({g['weight']:.1f} lb/ft)",
        ] + ([
            f"  Composite:  {g['stud_count']} studs total "
            f"({g['stud_count']//2} per side)  |  Ieff: {g['Ieff']:.0f} in4",
        ] if g.get('composite') else []) + [
            f"  Point loads: {g['n_point_loads']} x "
            f"{g['point_load_kips']:.1f} kips (factored)",
            f"  Equiv DL:   {g['equiv_DL']:.4f} kip/ft  |  "
            f"Equiv LL: {g['equiv_LL']:.4f} kip/ft",
            f"  Reaction:   {g['reaction']:.1f} kips each end (factored)",
            f"  Status:     {'PASS' if g['passes'] else 'FAIL'}",
            "",
            f"COLUMN ({c['height']:g}ft, K={c['K']})",
            f"  Section: {c['name']}  ({c['weight']:.1f} lb/ft)",
            f"  Pu:      {c['Pu']:.1f} kips "
            f"({self.num_floors} floor{'s' if self.num_floors > 1 else ''})",
            f"  Mu:      {c['Mu']:.1f} kip-ft (nominal eccentricity)",
            f"  Status:  {'PASS' if c['passes'] else 'FAIL'}",
            "",
            f"TOTAL STEEL: {self.results['total_weight']:,.0f} lbs",
            f"  Beams:  {beam_wt:,.0f} lbs  "
            f"({b['num_beams']} x {b['weight']:.1f} x {b['span']:g}ft)",
            f"  Girder: {girder_wt:,.0f} lbs  "
            f"(1 x {g['weight']:.1f} x {g['span']:g}ft)",
            f"  Column: {column_wt:,.0f} lbs  "
            f"(1 x {c['weight']:.1f} x {c['height']:g}ft)",
        ] + ([
            "",
            "COMPOSITE SAVINGS",
            f"  Beam steel (composite):       {savings.get('beam_composite_lbs', 0):,.0f} lbs",
            f"  Beam steel (non-composite):   {savings.get('beam_noncomp_lbs', 0):,.0f} lbs",
            f"  Beam savings:                 {savings.get('beam_savings_lbs', 0):,.0f} lbs",
            "",
            f"  Girder steel (composite):     {savings.get('girder_composite_lbs', 0):,.0f} lbs",
            f"  Girder steel (non-composite): {savings.get('girder_noncomp_lbs', 0):,.0f} lbs",
            f"  Girder savings:               {savings.get('girder_savings_lbs', 0):,.0f} lbs",
            "",
            f"  Total savings vs non-composite: {savings.get('total_savings_lbs', 0):,.0f} lbs",
            f"  Cost savings: ${savings.get('cost_savings', 0):,.0f}",
        ] if savings else []) + [
            "",
            "SLAB",
            f"  Deck: {slab.get('deck_type', 'N/A')}  |  Total thickness: {slab.get('total_thickness_in', 0):.1f} in",
            f"  Reinforcement: {slab.get('WWF_designation', 'N/A')}  |  Fire rating: {slab.get('fire_rating_hrs', 0)} hr",
            f"  Status: {'PASS' if slab.get('passes') else 'FAIL'}",
            "",
            "FOUNDATION",
            f"  Footing: {footing.get('B_ft', 0):.1f}ft x {footing.get('L_ft', 0):.1f}ft x {footing.get('h_in', 0):.0f}in",
            f"  Reinforcement: {footing.get('bar_size', '#5')} @ {footing.get('bar_spacing_in', 0):.1f} in",
            f"  Base plate: {base_plate.get('N_in', 0):.0f} x {base_plate.get('B_in', 0):.0f} x {base_plate.get('t_in', 0):.2f} in",
            "",
            f"ALL CHECKS: {status}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":

    print("Floor System Design Tests")
    print("=" * 60)

    # Test 1: Office floor
    print("\nTest 1: Office floor")
    print("40ft x 30ft bay, 50 psf DL, 80 psf LL, 10ft spacing")
    print("-" * 60)
    fs1 = FloorSystem(
        bay_length_ft=40, bay_width_ft=30,
        dead_load_psf=50, live_load_psf=80,
        beam_spacing_ft=10, num_floors=1, floor_height_ft=14)
    fs1.design_all()
    print(fs1.get_summary())

    # Test 2: Heavy industrial floor
    print("\n\nTest 2: Industrial floor")
    print("30ft x 20ft bay, 80 psf DL, 150 psf LL, 5ft spacing")
    print("-" * 60)
    fs2 = FloorSystem(
        bay_length_ft=30, bay_width_ft=20,
        dead_load_psf=80, live_load_psf=150,
        beam_spacing_ft=5, num_floors=1, floor_height_ft=16)
    fs2.design_all()
    print(fs2.get_summary())

    # Test 3: Multi-floor office
    print("\n\nTest 3: Multi-floor office (8 stories)")
    print("30ft x 25ft bay, 50 psf DL, 50 psf LL, 8ft spacing, 8 floors")
    print("-" * 60)
    fs3 = FloorSystem(
        bay_length_ft=30, bay_width_ft=25,
        dead_load_psf=50, live_load_psf=50,
        beam_spacing_ft=8, num_floors=8, floor_height_ft=14)
    fs3.design_all()
    print(fs3.get_summary())
