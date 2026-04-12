"""
AISC W-Shapes Database
Loads from SkyCiv CSV with all derived properties already calculated
"""

from pathlib import Path

import pandas as pd

STEEL_E  = 29000  # ksi
STEEL_FY = 50     # ksi

_DEFAULT_CSV = Path(__file__).resolve().parent / "Data for beam sizing - Sheet1.csv"


def load_beams(csv_path=None):
    if csv_path is None:
        csv_path = _DEFAULT_CSV
    df = pd.read_csv(csv_path, header=2)
    
    # Rename to clean internal names (CSV row 3: first col is unnamed; Zp header may contain newline)
    df = df.rename(columns={
        'Unnamed: 0':            'name',
        'd (in)':                'd',
        'bt (in)':               'bf',
        'Weight':                'weight',
        'tt (in)':               'tf',
        'Sx':                    'Sx',
        'tw (in)':               'tw',
        'r (in)':                'r',
        'Sy':                    'Sy',
        'rx':                    'rx',
        'A (in2)':               'A',
        'ry':                    'ry',
        'Iyp (in4)':             'Iy',
        'Izp (in4)':             'Ix',
        'Plastic\nMod Zp (in3)': 'Zx',
    })
    
    # Keep only the columns we need
    keep = ['name', 'd', 'bf', 'tf', 'tw', 'r', 'A',
            'weight', 'Ix', 'Iy', 'Sx', 'Sy', 'rx', 'ry', 'Zx']
    df = df[keep]
    
    # Drop rows with missing critical values
    critical = ['d', 'bf', 'tf', 'tw', 'r', 'A', 'Ix', 'Zx', 'weight']
    df = df.dropna(subset=critical)
    df = df.sort_values('weight').reset_index(drop=True)
    
    return df


BEAMS_DF = load_beams()

# Column design uses W10, W12, W14 only (typical column lines); beams use full catalog
COLUMN_SECTIONS = BEAMS_DF[
    BEAMS_DF['name'].str.match(r'W(10|12|14)x\d+', na=False)
].reset_index(drop=True)


def _build_hss_database():
    profiles = [
        {"name": "HSS6x6x1/4", "B": 6.0, "t": 0.25},
        {"name": "HSS6x6x3/8", "B": 6.0, "t": 0.375},
        {"name": "HSS8x8x1/4", "B": 8.0, "t": 0.25},
        {"name": "HSS8x8x3/8", "B": 8.0, "t": 0.375},
        {"name": "HSS10x10x1/4", "B": 10.0, "t": 0.25},
        {"name": "HSS10x10x3/8", "B": 10.0, "t": 0.375},
    ]
    rows = []
    steel_density_lb_per_in3 = 0.2836
    for profile in profiles:
        B = profile["B"]
        t = profile["t"]
        Bi = B - 2 * t
        area = B**2 - max(Bi, 0)**2
        Ix = (B**4 - max(Bi, 0)**4) / 12.0
        ry = (Ix / area) ** 0.5 if area > 0 else 0.0
        weight = area * 12.0 * steel_density_lb_per_in3
        rows.append(
            {
                "name": profile["name"],
                "d": B,
                "bf": B,
                "tf": t,
                "tw": t,
                "A": round(area, 3),
                "Ix": round(Ix, 3),
                "Iy": round(Ix, 3),
                "rx": round(ry, 3),
                "ry": round(ry, 3),
                "weight": round(weight, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("A").reset_index(drop=True)


HSS_DF = _build_hss_database()


def get_beam_names():
    return list(BEAMS_DF['name'])

def get_num_beams():
    return len(BEAMS_DF)

def get_beam_by_index(index):
    row  = BEAMS_DF.iloc[index]
    name = row['name']
    props = row.to_dict()
    return name, props


def get_num_columns():
    return len(COLUMN_SECTIONS)


def get_column_by_index(index):
    row  = COLUMN_SECTIONS.iloc[index]
    name = row['name']
    props = row.to_dict()
    return name, props


def get_num_hss():
    return len(HSS_DF)


def get_hss_by_index(index):
    row = HSS_DF.iloc[index]
    name = row["name"]
    props = row.to_dict()
    return name, props


if __name__ == "__main__":
    print(f"Loaded {get_num_beams()} beams")
    print(f"\nColumns: {list(BEAMS_DF.columns)}")
    print(f"\nFirst beam:")
    print(BEAMS_DF.iloc[0])
    
    print("\nSanity check — weight vs beam name:")
    passed = 0
    failed = 0
    for _, row in BEAMS_DF.iterrows():
        try:
            expected = float(row['name'].split('x')[1])
            actual   = round(row['weight'], 0)
            if abs(actual - expected) > 3:
                print(f"  MISMATCH: {row['name']} → calculated {actual}, expected {expected}")
                failed += 1
            else:
                passed += 1
        except:
            pass
    print(f"  {passed} passed, {failed} mismatched")
