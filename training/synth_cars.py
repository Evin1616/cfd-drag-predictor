"""Parametric car-like bodies with physics-inspired drag and lift labels.

This is the FALLBACK dataset, used because DrivAerNet++ could not be downloaded in
the build environment. The labels are NOT CFD results. They come from an
empirical model built around the published Ahmed-body trends (Ahmed et al.,
1984): drag dips for shallow rear slants, peaks near a 30 degree slant and drops
once the flow fully separates; lift rises with slant angle. On top of that we add
nose bluntness, boat-tail taper, ride height, section boxiness, wheel and skin
friction terms. It is good enough to show that a point-cloud network can learn a
shape -> coefficient mapping, and the same training code runs on DrivAerNet++
(see drivaernet.py) once the real data is available.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import trimesh

# Ahmed-like trend tables, rear slant angle in degrees.
_PHI = np.array([0, 5, 10, 15, 20, 25, 28, 30, 32, 35, 40], dtype=float)
_CD_PHI = np.array([0.300, 0.284, 0.274, 0.272, 0.284, 0.312, 0.345, 0.372, 0.330, 0.300, 0.296])
_CL_PHI = np.array([0.040, 0.070, 0.105, 0.150, 0.205, 0.270, 0.320, 0.345, 0.200, 0.140, 0.120])


@dataclass
class CarParams:
    length_m: float      # physical length, only used to give the STL a real size
    width: float         # all remaining lengths are fractions of the car length
    height: float
    clearance: float
    nose_len: float      # x where the hood reaches full height
    nose_round: float    # 0 = blunt box nose, 1 = fully rounded
    hood: float          # hood height as a fraction of body height
    roof_start: float    # x where the windshield meets the roof
    slant_deg: float     # rear slant angle (0 = squareback)
    slant_len: float     # streamwise length of the rear slant
    taper: float         # boat-tail: rear width/height reduction
    boxiness: float      # superellipse exponent of the cross-section
    wheel_r: float

    @staticmethod
    def random(rng: np.random.Generator) -> "CarParams":
        return CarParams(
            length_m=rng.uniform(3.8, 5.2),
            width=rng.uniform(0.36, 0.46),
            height=rng.uniform(0.26, 0.36),
            clearance=rng.uniform(0.02, 0.05),
            nose_len=rng.uniform(0.14, 0.26),
            nose_round=rng.uniform(0.0, 1.0),
            hood=rng.uniform(0.50, 0.72),
            roof_start=rng.uniform(0.36, 0.46),
            slant_deg=rng.uniform(0.0, 40.0),
            slant_len=rng.uniform(0.15, 0.32),
            taper=rng.uniform(0.0, 0.12),
            boxiness=rng.uniform(2.5, 6.0),
            wheel_r=rng.uniform(0.065, 0.085),
        )


def _profiles(p: CarParams, x: np.ndarray):
    """Top, bottom and half-width of the body along x (nose at x = 0)."""
    top_full = p.clearance + p.height
    hood_z = p.clearance + p.hood * p.height
    t = np.clip(x / p.nose_len, 0, 1)
    nose_top = hood_z - (hood_z - p.clearance - 0.35 * p.height) * (1 - t) ** 2
    wind = np.clip((x - p.nose_len) / (p.roof_start - p.nose_len), 0, 1)
    top = np.where(x < p.nose_len, nose_top, hood_z + (top_full - hood_z) * np.sin(wind * np.pi / 2))
    # rear slant, but never lower than a sensible rear deck height
    xs = 1.0 - p.slant_len
    drop = np.clip(x - xs, 0, None) * np.tan(np.radians(p.slant_deg))
    deck = p.clearance + 0.45 * p.height
    top = np.maximum(top - drop, np.minimum(top, deck))

    bottom = np.full_like(x, p.clearance)
    bottom += 0.25 * p.height * np.clip(1 - x / (0.6 * p.nose_len), 0, 1) ** 2  # nose undercut
    rear = np.clip((x - 0.85) / 0.15, 0, 1)
    bottom += 0.5 * p.taper * p.height * rear  # diffuser / boat tail from below
    top -= 0.5 * p.taper * p.height * rear

    half_w = np.full_like(x, p.width / 2)
    half_w *= 1 - 0.18 * np.clip(1 - x / p.nose_len, 0, 1) ** 2  # plan-view nose rounding
    half_w *= 1 - p.taper * rear

    # nose rounding shrinks the section towards a point as x -> 0
    s = np.ones_like(x)
    near = x < p.nose_len
    u = 1 - x[near] / p.nose_len
    s[near] = 1 - p.nose_round * (1 - np.sqrt(np.clip(1 - u**2, 0, 1)))
    s = np.maximum(s, 0.04)
    zc = 0.5 * (top + bottom)
    return zc, 0.5 * (top - bottom) * s, half_w * s


def build_body(p: CarParams, nx: int = 90, nt: int = 64) -> trimesh.Trimesh:
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nx)))  # cluster stations at both ends
    zc, hz, hy = _profiles(p, x)
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    e = 2.0 / p.boxiness
    cy = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** e
    cz = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** e

    rings = np.stack([
        np.broadcast_to(x[:, None], (nx, nt)),
        hy[:, None] * cy[None, :],
        zc[:, None] + hz[:, None] * cz[None, :],
    ], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(nx - 1):
        a = i * nt + np.arange(nt)
        b = i * nt + (np.arange(nt) + 1) % nt
        faces.append(np.stack([a, a + nt, b], 1))
        faces.append(np.stack([b, a + nt, b + nt], 1))
    verts = [rings]
    # end caps (triangle fans)
    for idx, xi, flip in ((0, x[0], True), (nx - 1, x[-1], False)):
        c = len(rings) + (0 if idx == 0 else 1)
        ring = idx * nt + np.arange(nt)
        nxt = idx * nt + (np.arange(nt) + 1) % nt
        f = np.stack([np.full(nt, c), ring, nxt], 1)
        faces.append(f[:, [0, 2, 1]] if not flip else f)
    verts.append(np.array([[x[0], 0, zc[0]], [x[-1], 0, zc[-1]]]))
    mesh = trimesh.Trimesh(np.concatenate(verts), np.concatenate(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def build_wheels(p: CarParams) -> list[trimesh.Trimesh]:
    wheels = []
    wy = p.width / 2 - 0.035
    for wx in (0.18, 0.80):
        for side in (-1, 1):
            w = trimesh.creation.cylinder(radius=p.wheel_r, height=0.05, sections=32)
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            w.apply_translation([wx, side * wy, p.wheel_r])
            wheels.append(w)
    return wheels


def build_car(p: CarParams) -> trimesh.Trimesh:
    """Car mesh in physical metres, flow along +x, ground at z = 0."""
    mesh = trimesh.util.concatenate([build_body(p)] + build_wheels(p))
    mesh.apply_translation([-0.5, 0, 0])
    mesh.apply_scale(p.length_m)
    return mesh


def coefficients(p: CarParams, mesh: trimesh.Trimesh, frontal_area_m2: float,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Physics-inspired (not CFD) drag and lift coefficients for one car."""
    cd = np.interp(p.slant_deg, _PHI, _CD_PHI)
    cl = np.interp(p.slant_deg, _PHI, _CL_PHI)

    cd += 0.07 * (1 - p.nose_round) ** 1.5           # blunt nose -> stagnation drag
    cd -= 0.55 * p.taper * (1 - 2.5 * p.taper)        # boat tailing, diminishing returns
    cd += 1.0 * (p.clearance - 0.035)                  # higher ride height, more underbody drag
    cd += 0.012 * (p.boxiness - 4.0)                   # sharp section corners
    windshield = (p.height * (1 - p.hood)) / (p.roof_start - p.nose_len)
    cd += 0.04 * (windshield - 0.4)                    # upright windshield
    cd += 0.25 * (p.wheel_r - 0.075) / 0.075 * 0.1     # bigger wheels
    wet = mesh.area / frontal_area_m2
    cd += 0.0032 * wet                                 # skin friction on wetted area

    cl += 2.5 * (p.clearance - 0.035)                  # ground effect
    cl -= 0.6 * p.taper                                # diffuser downforce
    cl += 0.08 * p.nose_round - 0.04                   # rounded nose feeds more air under the car
    cl += 0.10 * (p.hood - 0.6)

    if rng is not None:  # small "solver noise"
        cd *= 1 + rng.normal(0, 0.01)
        cl += rng.normal(0, 0.01)
    return float(cd), float(cl)


def params_dict(p: CarParams) -> dict:
    return asdict(p)
