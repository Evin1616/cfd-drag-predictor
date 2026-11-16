"""Everything the app needs to turn an STL into predicted coefficients.

This is the only module the UI imports from cfd_core, so the training code and
weights can live in a private package as long as this interface stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh

from cfd_core.geometry import N_POINTS, auto_orient, canonicalize, frontal_area, load_mesh, sample_points
from cfd_core.model import PointNetRegressor

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "pointnet_cd_cl.pt"
RHO_AIR = 1.225  # kg/m^3 at sea level, 15 C


@dataclass
class Prediction:
    cd: float
    cl: float
    cd_spread: float          # std over point-cloud resamplings (sampling sensitivity, not a CFD error bar)
    cl_spread: float
    frontal_area_m2: float
    length_m: float
    mesh: trimesh.Trimesh     # in metres, flow along +x, ground at z = 0
    points: np.ndarray        # canonical point cloud the model saw (N, 6)
    out_of_range: list[str]   # shape properties outside the training distribution

    def forces(self, velocity_ms: float) -> tuple[float, float]:
        q = 0.5 * RHO_AIR * velocity_ms**2
        return q * self.cd * self.frontal_area_m2, q * self.cl * self.frontal_area_m2


class Predictor:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "cpu"):
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        self.model = PointNetRegressor(**ckpt["config"]).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.mean = np.asarray(ckpt["target_mean"], np.float32)
        self.std = np.asarray(ckpt["target_std"], np.float32)
        self.meta = ckpt.get("meta", {})
        self.ranges = ckpt.get("aspect_ranges", {})
        self.device = device
        # older checkpoints have no test metrics; a drag-only model has no "cl" entry
        self.has_lift = "cl" in self.meta.get("test_metrics", {"cl": None})

    def predict_file(self, path, units_scale: float = 1.0, auto_orient_mesh: bool = True,
                     flip_x: bool = False, n_samples: int = 8) -> Prediction:
        mesh = load_mesh(path)
        mesh.apply_scale(units_scale)
        if auto_orient_mesh:
            mesh = auto_orient(mesh)
        return self.predict_mesh(mesh, flip_x=flip_x, n_samples=n_samples)

    def predict_mesh(self, mesh: trimesh.Trimesh, flip_x: bool = False, n_samples: int = 8) -> Prediction:
        if flip_x:
            mesh = mesh.copy()
            mesh.apply_transform(np.diag([-1.0, -1.0, 1.0, 1.0]))
        lo, hi = mesh.bounds
        mesh = mesh.copy()
        mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        length = float(mesh.extents[0])
        area = frontal_area(mesh)

        canon = canonicalize(mesh)
        batch = np.stack([sample_points(canon, N_POINTS, seed=s) for s in range(n_samples)])
        with torch.no_grad():
            out = self.model(torch.from_numpy(batch).to(self.device)).cpu().numpy()
        out = out * self.std + self.mean
        if not self.has_lift:
            out[:, 1] = np.nan
        return Prediction(
            cd=float(out[:, 0].mean()), cl=float(out[:, 1].mean()),
            cd_spread=float(out[:, 0].std()), cl_spread=float(out[:, 1].std()),
            frontal_area_m2=area, length_m=length, mesh=mesh, points=batch[0],
            out_of_range=self._check_range(canon),
        )

    def _check_range(self, canon: trimesh.Trimesh) -> list[str]:
        ext = canon.extents
        warnings = []
        for name, value in (("width/length", ext[1]), ("height/length", ext[2])):
            lo, hi = self.ranges.get(name, (-np.inf, np.inf))
            if not lo * 0.9 <= value <= hi * 1.1:
                warnings.append(f"{name} = {value:.2f} (training range {lo:.2f}-{hi:.2f})")
        return warnings
