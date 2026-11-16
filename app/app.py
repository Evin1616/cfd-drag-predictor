"""Gradio front end: drop in an STL, get a 3D view and predicted drag and lift.

    python -m app.app            # then open http://127.0.0.1:7860

The UI only talks to cfd_core.inference.Predictor.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from cfd_core.inference import Predictor

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*.stl"))
UNITS = {"metres": 1.0, "millimetres": 1e-3, "centimetres": 1e-2, "inches": 0.0254}

predictor = Predictor(os.environ.get("CFD_WEIGHTS", ROOT / "weights" / "pointnet_cd_cl.pt"))


def mesh_figure(pred) -> go.Figure:
    m = pred.mesh
    v, f = m.vertices, m.faces
    # colour faces by how directly they face the oncoming flow: a rough
    # stagnation-pressure picture (Newtonian Cp ~ cos^2), not a CFD result
    n = m.face_normals
    cp = np.where(n[:, 0] < 0, n[:, 0] ** 2, 0.0)
    fig = go.Figure(go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
        intensity=cp, intensitymode="cell", colorscale="Turbo", cmin=0, cmax=1,
        colorbar=dict(title="frontal<br>exposure", thickness=12, len=0.6),
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6),
        lightposition=dict(x=-2000, y=-1000, z=3000), flatshading=False,
        hovertemplate="x %{x:.2f} m<br>y %{y:.2f} m<br>z %{z:.2f} m<extra></extra>",
    ))
    L = pred.length_m
    fig.add_trace(go.Cone(x=[-0.75 * L], y=[0], z=[0.3 * m.extents[2]], u=[1], v=[0], w=[0],
                          sizemode="absolute", sizeref=0.12 * L, anchor="tip", showscale=False,
                          colorscale=[[0, "#4a90d9"], [1, "#4a90d9"]], hoverinfo="skip", name="flow"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=460, paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(aspectmode="data", xaxis_title="x (flow)", yaxis_title="y", zaxis_title="z",
                   camera=dict(eye=dict(x=-1.4, y=-1.5, z=0.8))),
    )
    return fig


def speed_figure(pred, v_now: float) -> go.Figure:
    v = np.linspace(0, 70, 141)
    drag, lift = pred.forces(v)
    fig = go.Figure()
    fig.add_scatter(x=v * 3.6, y=drag, name="Drag", line=dict(width=3, color="#d9534f"))
    if predictor.has_lift:
        fig.add_scatter(x=v * 3.6, y=lift, name="Lift", line=dict(width=3, color="#4a90d9"))
    d_now, _ = pred.forces(v_now)
    fig.add_scatter(x=[v_now * 3.6], y=[d_now], mode="markers", marker=dict(size=11, color="#d9534f"),
                    name="selected speed", showlegend=False)
    p_kw = d_now * v_now / 1000
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"Aero forces vs speed  ·  {p_kw:.1f} kW to overcome drag at {v_now * 3.6:.0f} km/h", font=dict(size=14)),
        xaxis_title="speed (km/h)", yaxis_title="force (N)", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return fig


def run(file, units, auto_orient, flip, speed_kmh):
    if file is None:
        raise gr.Error("Drop an .stl file first, or pick one of the examples.")
    path = file if isinstance(file, str) else file.name
    pred = predictor.predict_file(path, units_scale=UNITS[units], auto_orient_mesh=auto_orient, flip_x=flip)
    v = speed_kmh / 3.6
    drag, lift = pred.forces(v)

    def card(label, value, sub):
        return f"<div class='card'><div class='lbl'>{label}</div><div class='val'>{value}</div><div class='sub'>{sub}</div></div>"

    cards = "<div class='cards'>" + "".join([
        card("Drag coefficient C<sub>d</sub>", f"{pred.cd:.3f}", f"± {pred.cd_spread:.3f} resampling spread"),
        card("Lift coefficient C<sub>l</sub>", f"{pred.cl:.3f}", f"± {pred.cl_spread:.3f} resampling spread")
        if predictor.has_lift else card("Drag power", f"{drag * v / 1000:,.1f} kW", f"at {speed_kmh:.0f} km/h"),
        card("Drag force", f"{drag:,.0f} N", f"at {speed_kmh:.0f} km/h"),
        card("Frontal area", f"{pred.frontal_area_m2:.2f} m²", f"length {pred.length_m:.2f} m"),
    ]) + "</div>"
    if pred.out_of_range:
        cards += ("<div class='warn'>This shape is outside what the model was trained on, so treat the "
                  "numbers with caution: " + "; ".join(pred.out_of_range) + "</div>")
    if not 2.0 <= pred.length_m <= 7.0:
        cards += (f"<div class='warn'>The body is {pred.length_m:.2f} m long. If that is not car-sized, "
                  "check the units setting.</div>")
    return mesh_figure(pred), cards, speed_figure(pred, v)


CSS = """
.cards {display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px;}
.card {border:1px solid var(--border-color-primary); border-radius:12px; padding:14px 16px; background:var(--block-background-fill);}
.card .lbl {font-size:0.85rem; opacity:0.75;}
.card .val {font-size:1.9rem; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums;}
.card .sub {font-size:0.78rem; opacity:0.6; margin-top:2px;}
.warn {margin-top:12px; padding:10px 14px; border-radius:10px; background:rgba(240,173,78,0.15); border:1px solid rgba(240,173,78,0.5);}
"""

DATASET = predictor.meta.get("dataset", "unknown")
TEST = predictor.meta.get("test_metrics", {})
ACC = (f"Held-out test set: C<sub>d</sub> R² {TEST['cd']['r2']:.3f}, mean error {TEST['cd']['mape_pct']:.1f}%"
       if TEST else "")

with gr.Blocks(title="AeroSurrogate") as demo:
    gr.Markdown(
        f"# AeroSurrogate: instant {'drag and lift' if predictor.has_lift else 'drag'} from an STL\n"
        "A PointNet deep-learning surrogate that predicts aerodynamic coefficients from a car's surface "
        "geometry in well under a second, instead of hours of CFD."
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            stl = gr.File(label="Drop an .stl file", file_types=[".stl"])
            units = gr.Dropdown(list(UNITS), value="metres", label="STL units")
            auto = gr.Checkbox(value=True, label="Auto-orient (longest axis = flow direction, shortest = up)")
            flip = gr.Checkbox(value=False, label="Flip front/back (nose should point into the flow)")
            speed = gr.Slider(10, 250, value=110, step=5, label="Speed (km/h)")
            go_btn = gr.Button("Predict", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=stl, label="Example cars")
        with gr.Column(scale=2):
            view = gr.Plot(label="Geometry")
            cards = gr.HTML()
            chart = gr.Plot(label="Forces")
    gr.Markdown(
        f"<small>Model trained on: {DATASET}. {ACC}. Coordinate convention: flow along +x, z up, "
        "air density 1.225 kg/m³. The surface colouring is a simple frontal-exposure shading, not a "
        "predicted pressure field.</small>"
    )
    inputs = [stl, units, auto, flip, speed]
    go_btn.click(run, inputs, [view, cards, chart])
    stl.upload(run, inputs, [view, cards, chart])
    speed.release(run, inputs, [view, cards, chart])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="blue"), server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", 7860)))
