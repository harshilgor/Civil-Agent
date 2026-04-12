"""
Natural language parser for Civil Agent
Extracts structured design parameters from plain English input
Uses Claude API for flexible parsing
"""

import anthropic
import json
import re

from asce7_loads import auto_load_brief, list_supported_cities


def parse_with_claude(user_input):
    """
    Uses Claude API to extract structural parameters
    from natural language input.
    """
    client = anthropic.Anthropic()

    system_prompt = """You are a structural engineering \
parameter extractor. Extract design parameters from the \
user's message and return ONLY a JSON object with these fields:

{
  "span_ft":        number (beam span in feet),
  "dead_load":      number (dead load in kip/ft, default 1.0),
  "live_load":      number (live load in kip/ft, default 1.5),
  "point_load":     number (midspan point load in kips, default 0),
  "Lb_ft":          number or null (unbraced length in feet, \
null means fully braced),
  "defl_limit":     number (deflection limit divisor, default 360),
  "notes":          string (anything unclear or assumed)
}

Rules:
- If only one load is given assume it is total load, \
split 40% dead 60% live
- If "fully braced" or "braced" -> Lb_ft = 0
- If "unbraced" with no length -> Lb_ft = span_ft
- If deflection limit not mentioned -> 360
- Return ONLY the JSON, no other text"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}]
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r'```json|```', '', raw).strip()
    return json.loads(raw)


def parse_simple(user_input):
    """
    Rule-based fallback parser.
    Works without API key for basic inputs.
    """
    params = {
        'span_ft':          None,
        'dead_load':        1.0,
        'live_load':        1.5,
        'point_load':       0,
        'Lb_ft':            None,
        'defl_limit':       360,
        'notes':            '',
        'composite':        False,
        'slab_thickness':   3.5,
        'fc_ksi':           4.0,
        'beam_spacing':     10,
        'composite_ratio':  0.5,
        'city':             None,
        'occupancy':        None,
        'num_floors':       1,
        'floor_height_ft':  14.0,
        'bay_length_ft':    30.0,
        'bay_width_ft':     30.0,
        'loads_auto_calculated': False,
    }

    text = user_input.lower()

    span_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|\')', text)
    if span_match:
        params['span_ft'] = float(span_match.group(1))

    dead_match = re.search(
        r'(?:dead\s*(?:load)?\s*(?:of\s*)?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:kip(?:s)?(?:\s*/\s*ft)?\s*)?dead)',
        text)
    if dead_match:
        params['dead_load'] = float(dead_match.group(1) or dead_match.group(2))

    live_match = re.search(
        r'(?:live\s*(?:load)?\s*(?:of\s*)?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:kip(?:s)?(?:\s*/\s*ft)?\s*)?live)',
        text)
    if live_match:
        params['live_load'] = float(live_match.group(1) or live_match.group(2))

    # If no explicit dead/live loads were found, treat the first kip value as total load.
    if not dead_match and not live_match:
        load_match = re.search(
            r'(\d+(?:\.\d+)?)\s*kip', text)
        if load_match:
            total = float(load_match.group(1))
            params['dead_load'] = total * 0.4
            params['live_load'] = total * 0.6

    if 'fully braced' in text or 'fully-braced' in text:
        params['Lb_ft'] = 0
    elif 'unbraced' in text:
        ub_match = re.search(
            r'unbraced\s*(?:length\s*)?(?:of\s*)?(\d+(?:\.\d+)?)', text)
        params['Lb_ft'] = float(ub_match.group(1)) \
                          if ub_match else params['span_ft']

    if 'l/240' in text or '240' in text:
        params['defl_limit'] = 240
    elif 'l/480' in text or '480' in text:
        params['defl_limit'] = 480

    # Composite detection
    comp_keywords = ('composite', 'slab', 'shear stud', 'concrete deck',
                     'metal deck', 'shear connector')
    if any(kw in text for kw in comp_keywords):
        params['composite'] = True

    # Slab thickness: "3.5 inch slab" / "4 in slab"
    tc_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:in(?:ch)?|\")\s*slab', text)
    if tc_m:
        params['slab_thickness'] = float(tc_m.group(1))

    # Concrete strength: "fc=4ksi" / "4 ksi concrete" / "4000 psi"
    fc_m = re.search(r'fc\s*[=:]\s*(\d+(?:\.\d+)?)', text)
    if fc_m:
        params['fc_ksi'] = float(fc_m.group(1))
    else:
        psi_m = re.search(r'(\d{4,5})\s*psi', text)
        if psi_m:
            params['fc_ksi'] = float(psi_m.group(1)) / 1000.0

    # Beam spacing for composite: "beams at 10ft" / "10ft spacing"
    sp_m = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:ft|feet)?\s*(?:beam\s*)?spacing', text)
    if sp_m:
        params['beam_spacing'] = float(sp_m.group(1))

    # Composite ratio: "50% composite" / "75 percent"
    cr_m = re.search(r'(\d+)\s*(?:%|percent)\s*composite', text)
    if cr_m:
        params['composite_ratio'] = float(cr_m.group(1)) / 100.0

    floors_match = re.search(r'(\d+)\s*(?:story|storey|floor)', text)
    if floors_match:
        params['num_floors'] = int(floors_match.group(1))

    fh_match = re.search(r'floor\s*height\s*(?:of\s*)?(\d+(?:\.\d+)?)', text)
    if fh_match:
        params['floor_height_ft'] = float(fh_match.group(1))

    bay_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ft|feet)?\s*(?:x|by)\s*(\d+(?:\.\d+)?)', text)
    if bay_match:
        params['bay_length_ft'] = float(bay_match.group(1))
        params['bay_width_ft'] = float(bay_match.group(2))

    for occupancy in ["office", "retail", "warehouse", "residential", "hospital", "assembly", "parking", "education", "industrial"]:
        if occupancy in text:
            params["occupancy"] = occupancy
            break

    for city in list_supported_cities():
        if city.lower() in text:
            params["city"] = city
            break

    if params["city"] and params["occupancy"]:
        try:
            loads = auto_load_brief(
                city=params["city"],
                num_floors=params["num_floors"],
                floor_height_ft=params["floor_height_ft"],
                bay_length_ft=params.get("bay_length_ft", 30.0),
                bay_width_ft=params.get("bay_width_ft", 30.0),
                occupancy=params["occupancy"],
            )
            params.update(loads)
            params["loads_auto_calculated"] = True
        except Exception as exc:
            params["notes"] = f"{params['notes']} Auto-load generation unavailable: {exc}".strip()

    return params


def parse_column_simple(user_input):
    """Rule-based parser for column design queries."""
    params = {
        'type':       'column',
        'height_ft':  None,
        'axial_load': 200,
        'moment':     0,
        'K_factor':   1.0,
        'notes':      '',
    }
    text = user_input.lower()

    # Height: look for ft/feet near column/tall/high/height keywords
    ht = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|\')\s*(?:tall|high|height|column)?',
        text)
    if not ht:
        ht = re.search(
            r'(?:height|tall|high)\s*(?:of\s*)?(\d+(?:\.\d+)?)', text)
    if ht:
        params['height_ft'] = float(ht.group(1))

    # Axial load
    ax = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:kip(?:s)?)\s*(?:axial|compression|load)', text)
    if not ax:
        ax = re.search(
            r'(?:axial|compression)\s*(?:load\s*)?(?:of\s*)?(\d+(?:\.\d+)?)',
            text)
    if ax:
        params['axial_load'] = float(ax.group(1))

    # Moment
    mo = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:kip(?:s)?[- ]*(?:ft|feet)|k[- ]*ft)\s*(?:moment)?',
        text)
    if not mo:
        mo = re.search(
            r'moment\s*(?:of\s*)?(\d+(?:\.\d+)?)', text)
    if mo:
        params['moment'] = float(mo.group(1))

    # K factor
    kf = re.search(r'k\s*(?:=|factor)\s*(\d+(?:\.\d+)?)', text)
    if kf:
        params['K_factor'] = float(kf.group(1))

    return params


def _is_column_query(text):
    """Heuristic: does the input describe a column problem?"""
    col_words = ('column', 'axial', 'compression', 'tall', 'height')
    lower = text.lower()
    return any(w in lower for w in col_words)


def parse(user_input, use_claude=True):
    """
    Main parser -- tries Claude first, falls back to rule-based.
    Detects column vs beam automatically.
    """
    if use_claude:
        try:
            return parse_with_claude(user_input)
        except Exception as e:
            print(f"Claude parser failed ({e}), using simple parser")

    if _is_column_query(user_input):
        return parse_column_simple(user_input)
    return parse_simple(user_input)


if __name__ == "__main__":
    beam_tests = [
        "I need a beam for a 30ft span, 2 kip/ft dead load, 1.5 live load",
        "28 foot span, fully braced, total load 3 kips per foot",
        "size a beam for 25ft, unbraced length 12ft, live load 2.0",
        "40ft span office floor, L/360 deflection, 1.2 dead 1.8 live",
    ]
    col_tests = [
        "14ft tall column, 300 kips axial, 100 kip-ft moment",
        "20ft column, axial 200 kips, moment 150 kip-ft, K=1.0",
        "design a compression column 30ft height, 150 kips axial load",
    ]

    print("Parser Test")
    print("=" * 60)
    for test in beam_tests:
        print(f"\nInput:  {test}")
        result = parse(test, use_claude=False)
        print(f"Output: {json.dumps(result, indent=2)}")

    print("\n" + "=" * 60)
    print("Column Parser Test")
    print("=" * 60)
    for test in col_tests:
        print(f"\nInput:  {test}")
        result = parse(test, use_claude=False)
        print(f"Output: {json.dumps(result, indent=2)}")
