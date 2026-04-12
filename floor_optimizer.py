"""
Floor system optimizer - try multiple beam spacings,
find lightest total steel weight that passes all checks.
"""

from floor_system import FloorSystem


def optimize_floor(bay_length_ft, bay_width_ft,
                   dead_load_psf, live_load_psf,
                   spacings=None, num_floors=1,
                   floor_height_ft=14, defl_limit=360):
    """
    Try each beam spacing, return lightest passing total steel weight.

    Returns:
        best:         dict with keys spacing, total_weight, beam, girder, column, ...
        best_spacing: optimal beam spacing in ft
        all_results:  list of dicts for comparison table
    """
    if spacings is None:
        spacings = [5, 8, 10, 12, 15]

    results = []
    for spacing in spacings:
        n_beams = int(bay_length_ft / spacing) - 1
        if n_beams < 1 or spacing >= bay_width_ft:
            continue

        fs = FloorSystem(
            bay_length_ft, bay_width_ft,
            dead_load_psf, live_load_psf,
            beam_spacing_ft=spacing,
            num_floors=num_floors,
            floor_height_ft=floor_height_ft,
            defl_limit=defl_limit,
        )
        r = fs.design_all()
        results.append({
            'spacing':      spacing,
            'total_weight': r['total_weight'],
            'passes':       r['passes'],
            'beam':         r['beams']['name'],
            'girder':       r['girder']['name'],
            'column':       r['column']['name'],
            'results':      r,
            'summary':      fs.get_summary(),
        })

    passing = [r for r in results if r['passes']]
    if not passing:
        return None, None, results

    best = min(passing, key=lambda x: x['total_weight'])
    return best, best['spacing'], results


if __name__ == "__main__":

    print("Floor System Optimization")
    print("=" * 68)
    print("40ft x 30ft bay, 50 psf DL, 80 psf LL")
    print("Trying beam spacings: 5, 8, 10, 12, 15 ft")
    print()

    best, spacing, all_results = optimize_floor(
        bay_length_ft=40,
        bay_width_ft=30,
        dead_load_psf=50,
        live_load_psf=80,
    )

    print(f"{'Spacing':>10} {'Beam':>12} {'Girder':>12} "
          f"{'Column':>12} {'Total lbs':>12} {'Passes':>8}")
    print("-" * 68)

    for r in all_results:
        marker = " <--" if r['spacing'] == spacing else ""
        print(f"{r['spacing']:>8}ft  "
              f"{r['beam']:>12}  "
              f"{r['girder']:>12}  "
              f"{r['column']:>12}  "
              f"{r['total_weight']:>10,.0f}  "
              f"{'YES' if r['passes'] else 'NO':>8}"
              f"{marker}")

    if best:
        print(f"\nOptimal spacing: {spacing}ft")
        print(f"Total steel: {best['total_weight']:,.0f} lbs")
        print()
        print(best['summary'])
