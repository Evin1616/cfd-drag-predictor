"""Geometry utilities: load an STL, put it in a canonical frame, sample a point cloud.

Convention used everywhere in this project:
    +x  = free-stream flow direction (the nose of the car points to -x)
    +z  = up, ground plane at z = 0
    length is scaled to 1 so the model only sees shape, not size.
"""
from __future__ import annotations

import numpy as np
import trimesh

N_POINTS = 1024  # 2048 is better on a GPU; 1024 keeps CPU training under an hour


def load_mesh(path_or_file) -> trimesh.Trimesh:
    mesh = trimesh.load(path_or_file, file_type="stl", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Could not read a triangle mesh from this STL file.")
    mesh.remove_unreferenced_vertices()
    if mesh.is_watertight and mesh.volume < 0:  # inward-facing normals (e.g. DrivAerNet++ STLs)
        mesh.invert()
    return mesh


def auto_orient(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Map the longest bounding-box axis to x and the shortest to z.

    Works for car-like bodies exported in an arbitrary axis convention.
    It cannot tell front from back; use `flip_x` for that.
    """
    mesh = mesh.copy()
    extents = mesh.extents
    order = np.argsort(extents)[::-1]  # longest, middle, shortest
    perm = np.zeros((4, 4))
    for new_axis, old_axis in enumerate(order):
        perm[new_axis, old_axis] = 1.0
    perm[3, 3] = 1.0
    if np.linalg.det(perm[:3, :3]) < 0:  # keep it a proper rotation
        perm[1] *= -1
    mesh.apply_transform(perm)
    # Up or down? A car's roof is a broad surface while only the tyre contact
    # patches reach the ground, so compare surface area in thin top/bottom slabs.
    z = mesh.triangles_center[:, 2]
    lo, hi = mesh.bounds[:, 2]
    slab = 0.03 * (hi - lo)
    top = mesh.area_faces[z > hi - slab].sum()
    bottom = mesh.area_faces[z < lo + slab].sum()
    if bottom > top:
        mesh.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))
    return mesh


def canonicalize(mesh: trimesh.Trimesh, flip_x: bool = False) -> trimesh.Trimesh:
    """Centre in x/y, put the lowest point on z = 0 and scale length to 1."""
    mesh = mesh.copy()
    if flip_x:
        mesh.apply_transform(np.diag([-1.0, -1.0, 1.0, 1.0]))
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    mesh.apply_scale(1.0 / max(mesh.extents[0], 1e-9))
    return mesh


def frontal_area(mesh: trimesh.Trimesh, resolution: int = 512, closed: bool | None = None) -> float:
    """Projected area on the y-z plane (what the oncoming flow sees)."""
    from PIL import Image, ImageDraw

    lo, hi = mesh.bounds
    span = max(hi[1] - lo[1], hi[2] - lo[2])
    if span <= 0:
        return 0.0
    scale = (resolution - 1) / span
    img = Image.new("1", (resolution, resolution), 0)
    draw = ImageDraw.Draw(img)
    # for a closed surface the upstream-facing triangles already cover the silhouette
    closed = mesh.is_watertight if closed is None else closed
    facing = mesh.face_normals[:, 0] < 0 if closed else slice(None)
    tri = (mesh.triangles[facing][:, :, 1:] - lo[1:]) * scale
    for t in tri:
        draw.polygon([tuple(p) for p in t], fill=1, outline=1)
    return float(np.asarray(img).sum() / scale**2)


def sample_points(mesh: trimesh.Trimesh, n: int = N_POINTS, seed: int | None = 0) -> np.ndarray:
    """Area-weighted surface samples with their normals, shape (n, 6)."""
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    normals = mesh.face_normals[face_idx]
    return np.concatenate([pts, normals], axis=1).astype(np.float32)
