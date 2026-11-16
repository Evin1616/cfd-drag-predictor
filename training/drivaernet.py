"""Convert a local copy of DrivAerNet++ into the same .npz format as build_dataset.py.

DrivAerNet++ (Elrefaie et al., 2024, CC BY-NC 4.0) is hosted on Harvard Dataverse:
https://dataverse.harvard.edu/dataverse/DrivAerNet . Download the STL meshes
(or a subset of them) and the drag CSV, then run:

    python -m training.drivaernet --stl-dir /path/to/DrivAerNetPlusPlus_STLs \
        --cd-csv /path/to/DrivAerNetPlusPlus_Cd_8k_Updated.csv \
        --out data/drivaernet.npz [--limit 1000]
    python -m training.train --data data/drivaernet.npz --dataset-name "DrivAerNet++ (RANS CFD)"

Lift is optional: pass --cl-csv with a CSV that has a design column and a lift
column. Missing lift values are stored as NaN and ignored by the loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfd_core.geometry import canonicalize, frontal_area, load_mesh, sample_points
from training.build_dataset import POINTS_PER_SHAPE


def _read(csv: str | None, value_col: str | None, design_col: str) -> dict[str, float]:
    if csv is None:
        return {}
    df = pd.read_csv(csv)
    col = value_col or next(c for c in df.columns if c != design_col)
    return dict(zip(df[design_col].astype(str), df[col].astype(float)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl-dir", required=True)
    ap.add_argument("--cd-csv", required=True)
    ap.add_argument("--cd-col", default="Drag_Value")
    ap.add_argument("--cl-csv")
    ap.add_argument("--cl-col")
    ap.add_argument("--design-col", default="ID")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="data/drivaernet.npz")
    args = ap.parse_args()

    cd = _read(args.cd_csv, args.cd_col, args.design_col)
    cl = _read(args.cl_csv, args.cl_col, args.design_col)
    files = sorted(p for p in Path(args.stl_dir).glob("*.stl") if p.stem in cd)[: args.limit]
    print(f"{len(files)} STL files with a drag value")

    points, cds, cls, areas, names = [], [], [], [], []
    for i, f in enumerate(files):
        try:
            mesh = load_mesh(f)  # DrivAerNet++ is already x = flow, z = up, in metres
        except Exception as e:  # noqa: BLE001
            print(f"skip {f.name}: {e}")
            continue
        areas.append(frontal_area(mesh))
        points.append(sample_points(canonicalize(mesh), POINTS_PER_SHAPE, seed=i))
        cds.append(cd[f.stem]), cls.append(cl.get(f.stem, np.nan)), names.append(f.stem)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(files)}", flush=True)

    np.savez_compressed(
        args.out, points=np.stack(points).astype(np.float16), cd=np.array(cds, np.float32),
        cl=np.array(cls, np.float32), frontal_area=np.array(areas, np.float32),
        params=np.array([json.dumps({"design": n}) for n in names]),
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
