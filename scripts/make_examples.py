"""Write a few hand-picked example cars to examples/ for the app.

    python -m scripts.make_examples
"""
from pathlib import Path

from training.synth_cars import CarParams, build_car

BASE = dict(length_m=4.6, width=0.40, height=0.31, clearance=0.03, nose_len=0.2, nose_round=0.6,
            hood=0.6, roof_start=0.42, slant_deg=15, slant_len=0.25, taper=0.04, boxiness=4.0, wheel_r=0.075)

CARS = {
    "aero_teardrop": dict(nose_round=1.0, slant_deg=12, slant_len=0.32, taper=0.11, boxiness=2.8, height=0.28),
    "fastback_28deg": dict(slant_deg=28, slant_len=0.30),
    "boxy_van": dict(nose_round=0.0, slant_deg=0, height=0.35, boxiness=6.0, hood=0.7, taper=0.0, clearance=0.045),
    "family_hatch": dict(length_m=4.2, slant_deg=35, slant_len=0.18, height=0.34, nose_round=0.5),
}

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "examples" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    for name, overrides in CARS.items():
        build_car(CarParams(**{**BASE, **overrides})).export(out / f"{name}.stl")
        print("wrote", out / f"{name}.stl")
