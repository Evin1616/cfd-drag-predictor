"""Generate the synthetic car dataset as one compressed .npz file.

    python -m training.build_dataset --n 3000 --out data/synth_cars.npz
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from cfd_core.geometry import canonicalize, frontal_area, sample_points
from training.synth_cars import CarParams, build_car, coefficients, params_dict

POINTS_PER_SHAPE = 4096  # training subsamples 2048 of these each epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synth_cars.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    points, cd, cl, area, params = [], [], [], [], []
    t0 = time.time()
    for i in range(args.n):
        p = CarParams.random(rng)
        mesh = build_car(p)
        a = frontal_area(mesh)
        c_d, c_l = coefficients(p, mesh, a, rng)
        points.append(sample_points(canonicalize(mesh), POINTS_PER_SHAPE, seed=i))
        cd.append(c_d), cl.append(c_l), area.append(a), params.append(params_dict(p))
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{args.n} shapes  ({time.time() - t0:.0f}s)")

    np.savez_compressed(
        args.out,
        points=np.stack(points).astype(np.float16),
        cd=np.array(cd, np.float32),
        cl=np.array(cl, np.float32),
        frontal_area=np.array(area, np.float32),
        params=np.array([json.dumps(p) for p in params]),
    )
    print(f"saved {args.out}: Cd {np.min(cd):.3f}-{np.max(cd):.3f}, Cl {np.min(cl):.3f}-{np.max(cl):.3f}")


if __name__ == "__main__":
    main()
