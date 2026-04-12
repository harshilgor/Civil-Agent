"""
Civil Agent -- main interface
Combines parser + physics engine + RL agent for beams and columns
"""

from beams_data import (
    BEAMS_DF,
    COLUMN_SECTIONS,
    get_beam_by_index,
    get_column_by_index,
    get_num_beams,
    get_num_columns,
)
from beam_physics import calculate_demands, check_beam_design, explain_result
from connection_design import design_shear_tab
from column_physics import check_column_design, explain_column_result
from q_learning_agent import QLearningAgent
from train import run_training
from train_column import run_column_training
from parser import parse


class CivilAgent:

    def __init__(self):
        print("Loading Civil Agent...")
        print("Training beam RL agent...")
        self.rl_agent, _ = run_training(num_episodes=50000)
        print("Training column RL agent...")
        self.column_agent, _ = run_column_training(num_episodes=50000)
        print("Ready.\n")

    def find_optimal_beam(self, span_ft, dead_load, live_load,
                          Lb_ft=None, point_load=0, defl_limit=360,
                          composite=False, beam_spacing_ft=10,
                          slab_thickness_in=3.5, fc_ksi=4.0,
                          composite_ratio=0.5, shored=False):
        """
        Finds an efficient passing beam.

        The RL policy provides a fast first guess, but we always compare it
        against the lightest passing beam from a full scan. This avoids
        returning heavily over-designed sections when the RL table has not
        fully converged for a state.
        """
        if composite:
            from composite_beam import (
                design_composite_beam,
                explain_composite_result,
            )
            result = design_composite_beam(
                span_ft, dead_load, live_load,
                beam_spacing_ft=beam_spacing_ft,
                slab_thickness_in=slab_thickness_in,
                fc_ksi=fc_ksi,
                composite_ratio=composite_ratio,
                defl_limit=defl_limit,
                shored=shored,
            )
            if result and result.get('passes'):
                composite_result = dict(result)
                result['composite'] = composite_result
                result['explanation'] = explain_composite_result(result)
                result['beam_name'] = result.get('name', 'NONE')
                return result
            # Fall back to the non-composite scan when no composite beam passes.

        if Lb_ft is None:
            Lb_ft = span_ft

        state  = (span_ft, dead_load + live_load)
        rl_action = self.rl_agent.get_best_action(state)

        _, rl_beam = get_beam_by_index(rl_action)
        rl_passes, rl_weight, _, rl_details = check_beam_design(
            span_ft, dead_load, live_load, rl_beam,
            Lb_ft=Lb_ft, point_load=point_load,
            defl_limit=defl_limit
        )

        scan_beam, scan_action, scan_passes, scan_weight, scan_details = \
            self._scan_for_best(span_ft, dead_load, live_load,
                                Lb_ft, point_load, defl_limit)

        if scan_passes and (not rl_passes or scan_weight <= rl_weight):
            beam = scan_beam
            action = scan_action
            passes = scan_passes
            weight = scan_weight
            details = scan_details
        else:
            beam = rl_beam
            action = rl_action
            passes = rl_passes
            weight = rl_weight
            details = rl_details

        alternatives = self._get_alternatives(
            action, span_ft, dead_load, live_load,
            Lb_ft, point_load, defl_limit
        )

        beam_name = BEAMS_DF.iloc[action]['name']
        explanation = explain_result(
            beam_name, passes, details, alternatives)

        _, Vu, _ = calculate_demands(
            span_ft, dead_load, live_load, point_load)

        result = {
            'beam_name':   beam_name,
            'passes':      passes,
            'weight':      weight,
            'details':     details,
            'explanation': explanation,
            'Vu':          round(Vu, 1),
        }
        if passes:
            result['connection'] = design_shear_tab(Vu, beam)
        else:
            result['connection'] = None

        return result

    def debug_scan_prefix(self, span_ft, dead_load, live_load,
                          Lb_ft=None, point_load=0, defl_limit=360,
                          count=5):
        """Return the first few beams tried by the greedy scan."""
        if Lb_ft is None:
            Lb_ft = span_ft

        attempts = []
        for i in range(min(count, get_num_beams())):
            name, beam = get_beam_by_index(i)
            passes, weight, worst, details = check_beam_design(
                span_ft, dead_load, live_load, beam,
                Lb_ft=Lb_ft, point_load=point_load,
                defl_limit=defl_limit
            )
            attempts.append({
                'index': i,
                'beam_name': name,
                'passes': passes,
                'weight': weight,
                'worst_ratio': worst,
                'details': details,
            })
        return attempts

    def _scan_for_best(self, span_ft, dead_load, live_load,
                       Lb_ft, point_load, defl_limit):
        """Scans all beams lightest to heaviest, returns first pass."""
        for i in range(get_num_beams()):
            _, beam = get_beam_by_index(i)
            passes, weight, worst, details = check_beam_design(
                span_ft, dead_load, live_load, beam,
                Lb_ft=Lb_ft, point_load=point_load,
                defl_limit=defl_limit
            )
            if passes:
                return beam, i, passes, weight, details
        return None, 0, False, 0, {}

    def _get_alternatives(self, best_action, span_ft, dead_load,
                          live_load, Lb_ft, point_load, defl_limit):
        """Returns one lighter (fails) and one heavier (passes)."""
        alternatives = []

        if best_action > 0:
            _, lighter = get_beam_by_index(best_action - 1)
            p, w, _, _ = check_beam_design(
                span_ft, dead_load, live_load, lighter,
                Lb_ft=Lb_ft, point_load=point_load,
                defl_limit=defl_limit
            )
            alternatives.append(
                (BEAMS_DF.iloc[best_action-1]['name'], p, w))

        if best_action < get_num_beams() - 1:
            _, heavier = get_beam_by_index(best_action + 1)
            p, w, _, _ = check_beam_design(
                span_ft, dead_load, live_load, heavier,
                Lb_ft=Lb_ft, point_load=point_load,
                defl_limit=defl_limit
            )
            alternatives.append(
                (BEAMS_DF.iloc[best_action+1]['name'], p, w))

        return alternatives

    # --- Column methods --------------------------------------------------------

    def find_optimal_column(self, height_ft, Pu_kips, Mu_kipft, K=1.0):
        """
        Find lightest passing column using RL + greedy verification.
        """
        state = (height_ft, Pu_kips, Mu_kipft)
        rl_action = self.column_agent.get_best_action(state)

        _, rl_col = get_column_by_index(rl_action)
        rl_passes, rl_weight, rl_worst, rl_details = check_column_design(
            height_ft, Pu_kips, Mu_kipft, rl_col, K)

        scan_col, scan_action, scan_passes, scan_weight, scan_details = \
            self._scan_for_best_column(height_ft, Pu_kips, Mu_kipft, K)

        if scan_passes and (not rl_passes or scan_weight <= rl_weight):
            action  = scan_action
            passes  = scan_passes
            weight  = scan_weight
            details = scan_details
        else:
            action  = rl_action
            passes  = rl_passes
            weight  = rl_weight
            details = rl_details

        alternatives = self._get_column_alternatives(
            action, height_ft, Pu_kips, Mu_kipft, K)

        col_name = COLUMN_SECTIONS.iloc[action]['name']
        explanation = explain_column_result(
            col_name, passes, details, alternatives)

        return {
            'column_name': col_name,
            'passes':      passes,
            'weight':      weight,
            'details':     details,
            'explanation': explanation,
        }

    def _scan_for_best_column(self, height_ft, Pu_kips, Mu_kipft, K):
        for i in range(get_num_columns()):
            _, col = get_column_by_index(i)
            passes, weight, worst, details = check_column_design(
                height_ft, Pu_kips, Mu_kipft, col, K)
            if passes:
                return col, i, passes, weight, details
        return None, 0, False, 0, {}

    def _get_column_alternatives(self, best_action, height_ft,
                                 Pu_kips, Mu_kipft, K):
        alternatives = []
        if best_action > 0:
            _, lighter = get_column_by_index(best_action - 1)
            p, w, _, _ = check_column_design(
                height_ft, Pu_kips, Mu_kipft, lighter, K)
            alternatives.append(
                (COLUMN_SECTIONS.iloc[best_action - 1]['name'], p, w))
        if best_action < get_num_columns() - 1:
            _, heavier = get_column_by_index(best_action + 1)
            p, w, _, _ = check_column_design(
                height_ft, Pu_kips, Mu_kipft, heavier, K)
            alternatives.append(
                (COLUMN_SECTIONS.iloc[best_action + 1]['name'], p, w))
        return alternatives

    def respond_column(self, user_input):
        """Handle column design queries from natural language."""
        params = parse(user_input, use_claude=False)
        if params.get('type') != 'column' or params.get('height_ft') is None:
            return ("Could not parse column parameters. "
                    "Try: '14ft tall column, 300 kip axial, 100 kip-ft moment'")
        result = self.find_optimal_column(
            height_ft=params['height_ft'],
            Pu_kips=params['axial_load'],
            Mu_kipft=params['moment'],
            K=params.get('K_factor', 1.0),
        )
        d = result['details']
        def mark(r):
            return "OK" if r <= 1.0 else "FAIL"

        return (
            f"Optimal column: {result['column_name']}\n\n"
            f"Checks:\n"
            f"  Axial capacity:    {d['axial_ratio']:.2f}  {mark(d['axial_ratio'])}\n"
            f"  P-M interaction:   {d['pm_ratio']:.2f}  {mark(d['pm_ratio'])}  "
            f"({d['pm_equation']})\n"
            f"  Local buckl. (fl): {d['flange_ratio']:.2f}  {mark(d['flange_ratio'])}\n"
            f"  Local buckl. (web):{d['web_ratio']:.2f}  {mark(d['web_ratio'])}\n\n"
            f"{result['explanation']}"
        )

    # --- Beam respond -----------------------------------------------------------

    def respond(self, user_input):
        """
        Main entry point -- takes natural language,
        returns plain English response.
        """
        print(f"Parsing: {user_input}")
        params = parse(user_input, use_claude=False)

        if params['span_ft'] is None:
            return ("I couldn't find a span length in your message. "
                    "Try: '30ft span, 2 kip/ft dead load, 1.5 live load'")

        print(f"Parameters: {params}")

        result = self.find_optimal_beam(
            span_ft    = params['span_ft'],
            dead_load  = params['dead_load'],
            live_load  = params['live_load'],
            Lb_ft      = params['Lb_ft'],
            point_load = params['point_load'],
            defl_limit = params['defl_limit'],
        )

        Wu = 1.2 * params['dead_load'] + 1.6 * params['live_load']

        def mark(ratio):
            return "OK" if ratio <= 1.0 else "FAIL"

        d = result['details']
        response = (
            f"Optimal beam: {result['beam_name']}\n"
            f"\n"
            f"Factored load (LRFD): Wu = 1.2({params['dead_load']}) "
            f"+ 1.6({params['live_load']}) = {Wu:.2f} kip/ft\n"
            f"\n"
            f"Checks:\n"
            f"  Moment capacity:            {d['moment_ratio']:.2f}  "
            f"{mark(d['moment_ratio'])}\n"
            f"  Shear capacity:             {d['shear_ratio']:.2f}  "
            f"{mark(d['shear_ratio'])}\n"
            f"  Deflection (L/{params['defl_limit']}):          "
            f"{d['defl_ratio']:.2f}  {mark(d['defl_ratio'])}\n"
            f"  Lateral torsional buckling: {d['ltb_ratio']:.2f}  "
            f"{mark(d['ltb_ratio'])}  ({d.get('ltb_zone', '')})\n"
            f"  Local buckling (flange):    {d['flange_ratio']:.2f}  "
            f"{mark(d['flange_ratio'])}\n"
            f"\n"
            f"{result['explanation']}\n"
        )
        conn = result.get("connection")
        if conn:
            response += (
                f"\n\nShear tab connection (Vu = {result['Vu']} kips):\n"
                f"{conn['explanation']}"
            )
        response += (
            f"\n\nNotes: {params['notes'] if params['notes'] else 'None'}"
        )
        return response


class ConversationalCivilAgent(CivilAgent):
    """
    Civil Agent with lightweight conversation memory for iterative design work.
    """

    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.current_design = None
        self.current_brief = None

    def chat(self, user_message: str) -> str:
        intent = self._classify_intent(user_message)

        if intent == "new_design" or self.current_brief is None:
            brief = parse(user_message, use_claude=False)
            result = self.find_optimal_beam(
                span_ft=brief["span_ft"],
                dead_load=brief["dead_load"],
                live_load=brief["live_load"],
                Lb_ft=brief.get("Lb_ft"),
                point_load=brief.get("point_load", 0),
                defl_limit=brief.get("defl_limit", 360),
                composite=brief.get("composite", False),
                beam_spacing_ft=brief.get("beam_spacing", 10),
                slab_thickness_in=brief.get("slab_thickness", 3.5),
                fc_ksi=brief.get("fc_ksi", 4.0),
                composite_ratio=brief.get("composite_ratio", 0.5),
            )
            self.current_brief = brief
            self.current_design = result
            response = self._format_design_response(result, brief)
        elif intent == "what_if":
            modified = self._apply_modification(user_message, dict(self.current_brief))
            new_result = self.find_optimal_beam(
                span_ft=modified["span_ft"],
                dead_load=modified["dead_load"],
                live_load=modified["live_load"],
                Lb_ft=modified.get("Lb_ft"),
                point_load=modified.get("point_load", 0),
                defl_limit=modified.get("defl_limit", 360),
                composite=modified.get("composite", False),
                beam_spacing_ft=modified.get("beam_spacing", 10),
                slab_thickness_in=modified.get("slab_thickness", 3.5),
                fc_ksi=modified.get("fc_ksi", 4.0),
                composite_ratio=modified.get("composite_ratio", 0.5),
            )
            response = self._format_comparison(
                self.current_design,
                new_result,
                self.current_brief,
                modified,
            )
            self.current_brief = modified
            self.current_design = new_result
        elif intent == "explain":
            response = self._explain_current_design(user_message, self.current_design)
        elif intent == "compare":
            response = self._run_comparison(user_message, self.current_brief)
        else:
            response = self._answer_general_question(user_message)

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _classify_intent(self, message: str) -> str:
        lower = (message or "").lower()
        if any(term in lower for term in ("what if", "instead", "change", "increase", "reduce", "make it", "use ")):
            return "what_if"
        if any(term in lower for term in ("why", "explain", "how come", "governing")):
            return "explain"
        if any(term in lower for term in ("compare", "top 3", "alternatives", "options")):
            return "compare"
        if self.current_design is None:
            return "new_design"
        return "general"

    def _apply_modification(self, message: str, brief: dict):
        update = parse(message, use_claude=False)
        for key in ["span_ft", "dead_load", "live_load", "point_load", "Lb_ft", "defl_limit"]:
            value = update.get(key)
            if value is not None:
                brief[key] = value
        for key in ["composite", "beam_spacing", "slab_thickness", "fc_ksi", "composite_ratio"]:
            if key in update:
                brief[key] = update[key]
        return brief

    def _format_design_response(self, result: dict, brief: dict) -> str:
        return (
            f"Current design: {result['beam_name']}.\n"
            f"Span {brief['span_ft']} ft, DL {brief['dead_load']:.2f} kip/ft, "
            f"LL {brief['live_load']:.2f} kip/ft, L/{brief.get('defl_limit', 360)}.\n"
            f"Status: {'PASS' if result['passes'] else 'FAIL'} at max utilization "
            f"{result['details'].get('full_report', {}).get('controlling_ratio', result['details'].get('moment_ratio', 0)):.2f}.\n"
            f"{result['explanation']}"
        )

    def _format_comparison(self, old_result, new_result, old_brief, new_brief) -> str:
        old_name = old_result["beam_name"]
        new_name = new_result["beam_name"]
        old_wt = old_result["weight"]
        new_wt = new_result["weight"]
        delta = new_wt - old_wt
        return (
            f"Updated design: {old_name} -> {new_name}.\n"
            f"Previous loads/span: {old_brief['span_ft']} ft, {old_brief['dead_load']:.2f} DL, {old_brief['live_load']:.2f} LL.\n"
            f"New loads/span: {new_brief['span_ft']} ft, {new_brief['dead_load']:.2f} DL, {new_brief['live_load']:.2f} LL.\n"
            f"Weight change: {delta:+.1f} lb/ft. New result is {'PASS' if new_result['passes'] else 'FAIL'}.\n"
            f"{new_result['explanation']}"
        )

    def _explain_current_design(self, message: str, current_design: dict | None) -> str:
        if not current_design:
            return "There isn’t an active design yet. Give me a beam problem first and I’ll explain the result."
        report = current_design["details"].get("full_report", {})
        controlling = report.get("controlling_check", "unknown").replace("_", " ")
        ratio = report.get("controlling_ratio", current_design["details"].get("moment_ratio", 0))
        return (
            f"The current design uses {current_design['beam_name']} because it is the lightest passing option found in the scan/RL workflow.\n"
            f"The controlling check is {controlling} at {ratio:.2f} utilization.\n"
            f"{current_design['explanation']}"
        )

    def _run_comparison(self, message: str, brief: dict | None) -> str:
        if not brief:
            return "There isn’t an active design yet to compare. Start with a beam problem and then ask for alternatives."
        try:
            from section_recommender import predict_section

            recs = predict_section(
                span_ft=brief["span_ft"],
                dead_load=brief["dead_load"],
                live_load=brief["live_load"],
                beam_spacing=brief.get("beam_spacing", 10),
                top_k=3,
            )
        except Exception:
            recs = []
        lines = ["Top section options for the current problem:"]
        if not recs:
            lines.append("No ML model is available yet, so only the deterministic best design is currently active.")
        else:
            for name, confidence in recs:
                lines.append(f"- {name} ({confidence:.2f} confidence)")
        return "\n".join(lines)

    def _answer_general_question(self, message: str) -> str:
        if self.current_design:
            return (
                "I can help iterate from the current design. Ask me to change span, loads, bracing, or deflection limits, "
                "or ask why the current beam was chosen."
            )
        return self.respond(message)


if __name__ == "__main__":
    agent = CivilAgent()

    print("=" * 60)
    print("CIVIL AGENT -- Structural Beam Sizing")
    print("Type a beam sizing problem or 'quit' to exit")
    print("=" * 60)

    test_queries = [
        "30ft span, 1.2 kip/ft dead load, 1.8 live load, fully braced",
        "28 foot span, unbraced length 14ft, dead 1.0 live 2.0",
        "40ft span office floor, 1.5 dead 2.0 live, L/360 deflection",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        print(agent.respond(query))
        print()
