"""Stream a subset of DrivAerNet++ straight into the training .npz, one car at a time.

Each STL is ~76 MB, so instead of keeping the meshes this downloads one, samples
its point cloud, and deletes it. Finished cars are cached in --cache, so the run
can be interrupted and resumed. Each worker needs about 3 GB of RAM.

    python -m training.fetch_drivaernet --cd-csv data/DrivAerNetPlusPlus_Cd_8k_Updated.csv \
        --n 1500 --out data/drivaernet.npz

Source: the Hugging Face mirror of DrivAerNet++ (Elrefaie et al., 2024, CC BY-NC 4.0).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

from cfd_core.geometry import frontal_area, sample_points
from training.build_dataset import POINTS_PER_SHAPE

HF_URL = "https://huggingface.co/datasets/kasvii/drivaernet_plus/resolve/main/meshes/3DMeshesSTL/{}.stl"


def family(design: str) -> str:
    """'F_D_WM_WW_0123' -> 'F_D_WM_WW' (fastback/estate/notchback, smooth/detailed underbody)."""
    return design.rsplit("_", 1)[0]


def pick(designs: list[str], n: int, seed: int = 0) -> list[str]:
    """Stratified sample so every body family is represented in proportion."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"design": designs, "family": [family(d) for d in designs]})
    frac = min(1.0, n / len(df))
    out = df.groupby("family", group_keys=False).apply(
        lambda g: g.sample(max(1, round(len(g) * frac)), random_state=int(rng.integers(1 << 31))))
    return sorted(out["design"])[:n]


def load_stl_fast(path: Path) -> trimesh.Trimesh:
    """Read a binary STL without vertex merging (several times faster than trimesh.load on 1.5M faces).

    DrivAerNet++ meshes are closed but wound inside out, so flip them when the signed volume is negative.
    """
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    tri = np.fromfile(path, dtype=rec, offset=84)["v"].astype(np.float64)
    mesh = trimesh.Trimesh(tri.reshape(-1, 3), np.arange(3 * len(tri)).reshape(-1, 3), process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _limit_memory(gb: float) -> None:
    import resource
    limit = int(gb * 2**30)  # a runaway mesh then raises MemoryError instead of the OOM killer breaking the pool
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def process(design: str, cache: str, seed: int) -> str:
    out = Path(cache) / f"{design}.npz"
    if out.exists():
        return f"cached {design}"
    with tempfile.TemporaryDirectory() as tmp:
        stl = Path(tmp) / f"{design}.stl"
        subprocess.run(["curl", "-sSfL", "--retry", "4", "-m", "600", "-o", str(stl), HF_URL.format(design)], check=True)
        mesh = load_stl_fast(stl)  # DrivAerNet++ is already x = flow, z = up, in metres
        area, length = frontal_area(mesh, closed=True), mesh.extents[0]
        # same transform as canonicalize(), but in place: a 1.5M-face mesh copy costs >1 GB
        lo, hi = mesh.bounds
        mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        mesh.apply_scale(1.0 / length)
        np.savez(out, points=sample_points(mesh, POINTS_PER_SHAPE, seed=seed).astype(np.float16),
                 frontal_area=np.float32(area), length=np.float32(length))
    return f"done {design}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cd-csv", required=True)
    ap.add_argument("--design-col", default="ID")
    ap.add_argument("--cd-col", default="Drag_Value")
    ap.add_argument("--available", help="optional text file of STL names that exist on the mirror")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-gb", type=float, default=3.5, help="address-space cap per worker")
    ap.add_argument("--cache", default="data/drivaernet_cache")
    ap.add_argument("--out", default="data/drivaernet.npz")
    args = ap.parse_args()

    df = pd.read_csv(args.cd_csv)
    cd = dict(zip(df[args.design_col].astype(str), df[args.cd_col].astype(float)))
    designs = list(cd)
    if args.available:
        have = {Path(s.strip()).stem for s in open(args.available) if s.strip()}
        designs = [d for d in designs if d in have]
    chosen = pick(designs, args.n)
    print(f"{len(chosen)} of {len(designs)} designs selected", flush=True)

    Path(args.cache).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers, initializer=_limit_memory, initargs=(args.max_gb,),
                             max_tasks_per_child=1) as pool:
        futs = {pool.submit(process, d, args.cache, i): d for i, d in enumerate(chosen)}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
            except Exception as e:  # noqa: BLE001
                msg = f"skip {futs[f]}: {e}"
            print(f"[{k}/{len(chosen)}] {msg}", flush=True)

    points, cds, areas, names = [], [], [], []
    for d in chosen:
        f = Path(args.cache) / f"{d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        points.append(z["points"]), areas.append(z["frontal_area"]), cds.append(cd[d]), names.append(d)
    np.savez_compressed(
        args.out, points=np.stack(points), cd=np.array(cds, np.float32),
        cl=np.full(len(cds), np.nan, np.float32), frontal_area=np.array(areas, np.float32),
        params=np.array([json.dumps({"design": n}) for n in names]),
    )
    print(f"saved {args.out}: {len(names)} cars, Cd {min(cds):.3f}-{max(cds):.3f}")


if __name__ == "__main__":
    main()
