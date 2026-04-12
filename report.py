"""
PDF report generation for Civil Agent beam design output.
"""

from __future__ import annotations


def _pdf_output_bytes(pdf) -> bytes:
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")


def _pdf_safe_text(value: str) -> str:
    replacements = {
        "β": "beta",
        "φ": "phi",
        "√": "sqrt",
        "ε": "eps",
        "ₜ": "t",
        "ᵧ": "y",
        "â‰¥": ">=",
        "â‰¤": "<=",
        "â†’": "->",
        "Â·": "*",
        "Ã—": "x",
        "â€”": "-",
        "â€“": "-",
        "Â²": "^2",
        "â´": "^4",
        "′": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
    text = str(value)
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_rc_beam_report_pdf(result: dict) -> bytes:
    """
    Build a PDF report for ACI 318 reinforced-concrete beam checks.
    """
    from fpdf import FPDF

    section = result["section"]
    inputs = result["inputs"]
    flexure = result["flexure"]
    shear = result["shear"]

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 8, "Civil Agent - RC Beam Report", ln=True)
            self.ln(2)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Section Inputs", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5,
        f"b = {inputs['b_in']:.1f} in, h = {inputs['h_in']:.1f} in, d = {inputs['d_in']:.1f} in\n"
        f"f'c = {inputs['fc_psi']:.0f} psi, fy = {inputs['fy_psi']:.0f} psi\n"
        f"Tension steel = {inputs['tension_bar_count']} #{inputs['tension_bar_size']} bars "
        f"(As = {section.As:.2f} in^2)\n"
        f"Stirrups = {inputs['stirrup_legs']}-leg #{inputs['stirrup_bar_size']} @ {inputs['stirrup_spacing_in']:.1f} in"
        if inputs["stirrup_bar_size"] > 0 and inputs["stirrup_spacing_in"] > 0
        else "Stirrups = none\n"
        f"Mu = {inputs['Mu_kip_ft']:.2f} kip-ft, Vu = {inputs['Vu_kips']:.2f} kips",
    )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5,
        f"Overall status: {result['overall_status']}\n"
        f"Controlling check: {result['controlling_check']} at DCR = {result['controlling_dcr']:.3f}\n"
        f"Flexure: demand = {flexure.demand:.1f} lb-in, capacity = {flexure.capacity:.1f} lb-in, "
        f"DCR = {flexure.dcr:.3f}, status = {flexure.status.value.upper()}\n"
        f"Shear: demand = {shear.demand:.1f} lb, capacity = {shear.capacity:.1f} lb, "
        f"DCR = {shear.dcr:.3f}, status = {shear.status.value.upper()}",
    )
    pdf.ln(2)

    for label, check in [("Flexure", flexure), ("Shear", shear)]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, f"{label} Check", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"Code clause: {check.code_clause}", ln=True)
        for step in check.calc_steps:
            left_x = pdf.l_margin
            number_width = 16
            text_x = left_x + number_width
            text_width = pdf.w - pdf.r_margin - text_x

            pdf.set_font("Helvetica", "B", 8)
            pdf.set_x(left_x)
            pdf.cell(number_width, 4.5, f"{step.step_number}.", border=0)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_xy(text_x, pdf.get_y() - 4.5)
            pdf.multi_cell(
                text_width,
                4.5,
                _pdf_safe_text(
                    f"{step.description}\n"
                    f"{step.equation}\n"
                    f"{step.substituted}\n"
                    f"Result = {step.result} {step.unit}"
                    + (f"\nRef: {step.clause_ref}" if step.clause_ref else "")
                ),
            )
        if check.warnings:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, "Warnings", ln=True)
            pdf.set_font("Helvetica", "", 8)
            for warning in check.warnings:
                pdf.multi_cell(0, 4.5, _pdf_safe_text(f"- {warning}"))
        pdf.ln(2)

    return _pdf_output_bytes(pdf)

def generate_beam_report_pdf(
    result: dict,
    *,
    span_ft: float,
    dead_load: float,
    live_load: float,
    point_load: float,
    defl_limit: int,
    lb_display: float,
    Wu: float,
    Mu: float,
    Vu: float,
) -> bytes:
    """
    Build a simple PDF with beam checks and optional shear-tab connection section.
    """
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 8, "Civil Agent - Beam design report", ln=True)
            self.ln(2)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_compression(False)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Project inputs", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 5,
        f"Span: {span_ft:g} ft\n"
        f"Dead load: {dead_load:g} kip/ft  |  Live load: {live_load:g} kip/ft\n"
        f"Point load (midspan): {point_load:g} kips\n"
        f"Deflection limit: L/{defl_limit}\n"
        f"Unbraced length for LTB: {lb_display:g} ft\n"
        f"Wu = 1.2D + 1.6L = {Wu:.2f} kip/ft\n"
        f"Mu = {Mu:.2f} kip-ft  |  Vu = {Vu:.2f} kips",
    )
    pdf.ln(3)

    beam_name = result.get("beam_name", "")
    passes = result.get("passes", False)
    d = result.get("details") or {}
    full_report = d.get("full_report") or result.get("full_report") or {}

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Beam section", ln=True)
    pdf.set_font("Helvetica", "", 10)
    status = "PASS" if passes else "FAIL"
    pdf.cell(0, 5, f"Section: {beam_name}  ({status})", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Beam checks", ln=True)
    pdf.set_font("Helvetica", "", 9)

    if full_report:
        for check in full_report.get("checks", []):
            label = check.get("check", "").replace("_", " ").title()
            ratio = float(check.get("ratio", 0))
            status_text = "PASS" if check.get("passes") else "FAIL"
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(42, 5, label, border=0)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(18, 5, f"{ratio:.2f}", border=0)
            pdf.cell(28, 5, status_text, border=0)
            pdf.set_x(pdf.l_margin + 95)
            pdf.multi_cell(0, 5, check.get("equation", ""))
            note = check.get("note")
            if note:
                pdf.set_x(pdf.l_margin + 6)
                pdf.multi_cell(0, 4.5, f"Note: {note}")
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(
            0,
            5,
            f"Controlling check: {full_report.get('controlling_check', 'unknown').replace('_', ' ')} "
            f"at {float(full_report.get('controlling_ratio', 0)):.2f}",
            ln=True,
        )
        if full_report.get("fix_suggestion"):
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, f"Suggested fix: {full_report['fix_suggestion']}")
    else:
        def row_check(label: str, key: str, extra: str = ""):
            r = d.get(key)
            if r is None:
                return
            st = "PASS" if float(r) <= 1.0 else "FAIL"
            pdf.cell(60, 5, label, border=0)
            pdf.cell(25, 5, f"{float(r):.2f}", border=0)
            pdf.cell(20, 5, st + extra, ln=True)

        row_check("Moment", "moment_ratio")
        row_check("Shear", "shear_ratio")
        row_check("Deflection", "defl_ratio")
        row_check("LTB", "ltb_ratio", f" ({d.get('ltb_zone', '')})")
        fl = d.get("flange_ratio")
        wb = d.get("web_ratio")
        if fl is not None and wb is not None:
            lb_u = max(float(fl), float(wb))
            st_lb = "PASS" if lb_u <= 1.0 else "FAIL"
            pdf.cell(60, 5, "Local buckling", border=0)
            pdf.cell(25, 5, f"{lb_u:.2f}", border=0)
            pdf.cell(20, 5, st_lb, ln=True)

    pdf.ln(4)

    conn = result.get("connection")
    if conn is not None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "CONNECTION DESIGN", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 5, "Type: Shear tab (single plate)", ln=True)
        pdf.ln(1)

        n = conn["num_bolts"]
        t = conn["tab_thickness"]
        L = conn["tab_length"]
        from connection_design import BOLT_SPACING

        bl = conn.get("bolt_label", '3/4 in.')
        pdf.multi_cell(
            0, 5,
            f"Bolts:  {n} - {bl} A325-N bolts @ {BOLT_SPACING:g} in. spacing\n"
            f"Tab:    {t} in. x {L} in. A36 plate\n"
            f"Weld:   3/16 in. fillet weld both sides\n",
        )
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 5, "Check", border=0)
        pdf.cell(25, 5, "Ratio", border=0)
        pdf.cell(25, 5, "Status", ln=True)
        pdf.set_font("Helvetica", "", 9)

        chk = conn["checks"]
        for label, key in [
            ("Bolt shear", "bolt_shear"),
            ("Bolt bearing", "bolt_bearing"),
            ("Tab shear", "tab_shear"),
            ("Weld capacity", "weld"),
        ]:
            r = float(chk[key])
            st = "PASS" if r <= 1.0 else "FAIL"
            pdf.cell(50, 5, label, border=0)
            pdf.cell(25, 5, f"{r:.2f}", border=0)
            pdf.cell(25, 5, st, ln=True)

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        if conn.get("passes", False):
            pdf.cell(0, 6, "CONNECTION ADEQUATE", ln=True)
        else:
            pdf.cell(0, 6, "CONNECTION NOT ADEQUATE - review limits", ln=True)

        ctrl = max(conn["checks"], key=conn["checks"].get)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"Controlling: {ctrl.replace('_', ' ')}", ln=True)

    moment_conn = result.get("moment_connection")
    if moment_conn is not None:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "MOMENT CONNECTION DESIGN", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            0,
            5,
            f"Type: {moment_conn['type']}\n"
            f"Flange force: {moment_conn['flange_force_kips']:.1f} kips\n"
            f"Plate: {moment_conn['plate_width_in']:.1f} in x {moment_conn['plate_thickness_in']:.3f} in\n"
            f"Bolts per flange: {moment_conn['num_bolts_flange']} ({moment_conn['bolt_size']})\n"
            f"Weld size: {moment_conn['weld_size_in']:.3f} in\n"
            f"Panel zone: {'OK' if moment_conn.get('panel_zone_ok', False) else 'REVIEW'}\n"
            f"Estimated cost: ${moment_conn['cost_estimate']:,.0f}",
        )

    return _pdf_output_bytes(pdf)


def generate_optimization_report(results: dict, drawing_name: str) -> bytes:
    """
    Generate a PDF optimization report for drawing-review results.
    """
    from fpdf import FPDF

    summary = results.get("summary", {})

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 8, "Civil Agent Optimization Report", ln=True)
            self.set_font("Helvetica", "", 9)
            self.cell(0, 6, f"Drawing: {drawing_name}", ln=True)
            self.ln(2)

        def footer(self):
            self.set_y(-18)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "For preliminary review only", ln=True, align="C")
            self.cell(0, 5, f"Page {self.page_no()}", align="C")

    pdf = PDF(orientation="L", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5,
        f"Members reviewed: {summary.get('total_members', 0)}\n"
        f"Over-designed members: {summary.get('over_designed', 0)}\n"
        f"Current estimated steel: {summary.get('current_weight_lbs', 0):,.0f} lb\n"
        f"Optimized estimated steel: {summary.get('optimal_weight_lbs', 0):,.0f} lb\n"
        f"Potential steel savings: {summary.get('total_weight_saved_lbs', 0):,.0f} lb\n"
        f"Potential cost savings: ${summary.get('total_cost_saved', 0):,.0f}\n"
        f"Savings percent: {summary.get('savings_percent', 0):.1f}%",
    )
    pdf.ln(3)

    rows = []
    for member_type in ("beams", "girders"):
        for member in results.get(member_type, []) or []:
            rows.append(
                {
                    "mark": member.get("mark") or "",
                    "type": member_type[:-1].title(),
                    "current": member.get("current_section") or "",
                    "optimal": member.get("optimal_section") or "N/A",
                    "saved": float(member.get("weight_saved_per_ft", 0) or 0),
                    "status": member.get("status") or "Optimal",
                }
            )

    if rows:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Beam and Girder Recommendations", ln=True)
        pdf.set_font("Helvetica", "B", 9)
        widths = [26, 28, 42, 42, 28, 42]
        headers = ["Mark", "Type", "Current", "Optimal", "Saved", "Status"]
        for width, header in zip(widths, headers):
            pdf.cell(width, 6, header, border=1)
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for row in rows:
            values = [
                row["mark"],
                row["type"],
                row["current"],
                row["optimal"],
                f"{row['saved']:.1f}",
                row["status"],
            ]
            for width, value in zip(widths, values):
                pdf.cell(width, 6, str(value)[:24], border=1)
            pdf.ln()
        pdf.ln(3)

    columns = results.get("columns", []) or []
    if columns:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Column Notes", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for column in columns:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0,
                5,
                f"{column.get('mark') or 'Column'}: {column.get('current_section') or 'Unknown'}; "
                f"height {float(column.get('height_ft', 0) or 0):.1f} ft. "
                f"{column.get('note', '')}",
            )

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        0,
        5,
        "This report is for preliminary review only. A licensed Professional Engineer must verify loads, "
        "member unbraced lengths, framing assumptions, construction constraints, and all final designs.",
    )

    return _pdf_output_bytes(pdf)


def generate_3d_optimizer_report(
    building_data: dict,
    member_rows: list[dict],
    screenshot_path: str | None = None,
) -> bytes:
    """
    Generate a PDF report for the 3D optimizer page.
    """
    import os
    from fpdf import FPDF

    comp = building_data.get("before_after", {})

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 8, "Civil Agent 3D Optimizer Report", ln=True)
            self.set_font("Helvetica", "", 9)
            self.cell(
                0,
                6,
                f"{building_data.get('num_floors', 0)} floors | "
                f"{building_data.get('bays_x', 0)}x{building_data.get('bays_y', 0)} bays | "
                f"{building_data.get('bay_length_ft', 0):g} ft x {building_data.get('bay_width_ft', 0):g} ft",
                ln=True,
            )
            self.ln(2)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF(orientation="L", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5,
        f"Building type loads: DL {building_data.get('dead_psf', 0):g} psf | LL {building_data.get('live_psf', 0):g} psf\n"
        f"Before: {comp.get('before_lbs', 0):,.0f} lb | After: {comp.get('after_lbs', 0):,.0f} lb | "
        f"Saved: {comp.get('saved_lbs', 0):,.0f} lb ({comp.get('saved_pct', 0):.1f}%) | "
        f"Cost savings: ${comp.get('cost_savings', 0):,.0f}",
    )
    pdf.ln(2)

    if screenshot_path and os.path.exists(screenshot_path):
        try:
            pdf.image(screenshot_path, x=10, y=32, w=145)
            pdf.set_xy(162, 32)
        except Exception:
            pdf.set_x(pdf.l_margin)
    else:
        pdf.set_x(pdf.l_margin)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Representative Sections", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for label, base_key, opt_key in [
        ("Beam", "beams", "beams"),
        ("Girder", "girder", "girder"),
    ]:
        pdf.cell(
            0,
            5,
            f"{label}: {building_data['baseline'][base_key].get('name', 'NONE')} -> "
            f"{building_data['optimized'][opt_key].get('name', 'NONE')}",
            ln=True,
        )
    column_by_story = building_data.get("optimized", {}).get("column_by_story", [])
    if column_by_story:
        pdf.cell(0, 5, f"Top-floor column: {column_by_story[-1].get('name', 'NONE')}", ln=True)
        pdf.cell(0, 5, f"Ground-floor column: {column_by_story[0].get('name', 'NONE')}", ln=True)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 10)
    headers = ["Mark", "Type", "Floor", "Baseline", "Optimized", "Length", "Base U", "Opt U", "Saved lb"]
    widths = [22, 18, 24, 28, 28, 20, 18, 18, 22]
    for width, header in zip(widths, headers):
        pdf.cell(width, 6, header, border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for row in member_rows:
        values = [
            row.get("Mark", ""),
            row.get("Type", ""),
            row.get("Floor", ""),
            row.get("Baseline Section", ""),
            row.get("Optimized Section", ""),
            f"{float(row.get('Length (ft)', 0)):.1f}",
            f"{float(row.get('Baseline Utilization', 0)):.2f}",
            f"{float(row.get('Optimized Utilization', 0)):.2f}",
            f"{float(row.get('Total Saved (lb)', 0)):.0f}",
        ]
        for width, value in zip(widths, values):
            pdf.cell(width, 5, str(value)[:18], border=1)
        pdf.ln()

    return _pdf_output_bytes(pdf)


if __name__ == "__main__":
    print("report.py: use generate_beam_report_pdf() from app or tests.")
