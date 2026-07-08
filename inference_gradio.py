"""Gradio application for single-geometry GI-GNO inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr
import torch

from config import CHECKPOINT, GEOM_PARAM_FILE, NORM_STATS_FILE
from inference import load_model, run_inference

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _asset_label(path: str, loaded: bool) -> str:
    if loaded:
        return "Loaded"
    if Path(path).exists():
        return "Available"
    return "Missing"


def model_status() -> str:
    """Return current model asset status for the UI."""

    import inference

    loaded = inference._STATE is not None  # UI-only status display.
    ready = all(
        Path(path).exists()
        for path in (CHECKPOINT, NORM_STATS_FILE, GEOM_PARAM_FILE)
    )
    status = "Ready" if ready else "Missing assets"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    return (
        f"**Model Status:** {status}\n\n"
        f"**GPU:** {gpu}\n\n"
        f"**Checkpoint:** {_asset_label(CHECKPOINT, loaded)}\n\n"
        f"**Normalization:** {_asset_label(NORM_STATS_FILE, loaded)}\n\n"
        f"**Geometry Config:** {_asset_label(GEOM_PARAM_FILE, loaded)}"
    )


def _build_input_json(
    geom_name: str,
    alt_kft: str,
    reynolds: str,
    mach: str,
    alpha_deg: str,
    beta_deg: str,
) -> dict[str, Any]:
    return {
        "geom_name": geom_name,
        "alt_kft": alt_kft,
        "Re": reynolds,
        "M_inf": mach,
        "alpha_deg": alpha_deg,
        "beta_deg": beta_deg,
    }


def _format_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6e}"


def _summary_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    forces = result["forces"]
    return (
        "| Item | Value |\n"
        "| --- | ---: |\n"
        f"| Vertices | {result['vertices']} |\n"
        f"| Faces | {result['faces']} |\n"
        f"| Inference Time | {result['inference_time']:.6f} s |\n"
        f"| RMSE | {_format_number(metrics['rmse'])} |\n"
        f"| MAE | {_format_number(metrics['mae'])} |\n"
        f"| Correlation | {_format_number(metrics['corr'])} |\n"
        f"| CL | {_format_number(forces['CL'])} |\n"
        f"| CD | {_format_number(forces['CD'])} |"
    )


def simulate(
    vtk_file: str | None,
    geom_name: str,
    alt_kft: str,
    reynolds: str,
    mach: str,
    alpha_deg: str,
    beta_deg: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str, str | None, str | None, str]:
    """Run inference from UI inputs and return Gradio component updates."""

    try:
        if not vtk_file:
            return (
                "Upload a .vtk geometry before running simulation.",
                "",
                None,
                None,
                model_status(),
            )

        progress(0.10, desc="Validating inputs")
        input_json = _build_input_json(
            geom_name,
            alt_kft,
            reynolds,
            mach,
            alpha_deg,
            beta_deg,
        )

        progress(0.30, desc="Loading model")
        load_model()

        progress(0.55, desc="Running inference")
        result = run_inference(vtk_file, input_json)

        progress(0.90, desc="Saving prediction")
        if not result["success"]:
            return (
                result["error"],
                "",
                None,
                None,
                model_status(),
            )

        progress(1.0, desc="Complete")
        status = "Simulation completed successfully."
        return (
            status,
            _summary_markdown(result),
            result["output_vtk"],
            result["report"],
            model_status(),
        )
    except Exception as exc:  # noqa: BLE001 - keep UI from crashing.
        LOGGER.exception("UI simulation failed")
        return (f"Simulation failed: {exc}", "", None, None, model_status())


def build_app() -> gr.Blocks:
    """Create and return the Gradio application."""

    with gr.Blocks(
        title="GI-GNO CFD Inference",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css="""
        .container {max-width: 1120px; margin: auto;}
        .status-box textarea {font-size: 0.95rem;}
        """,
    ) as app:
        gr.Markdown("# GI-GNO CFD Inference")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown("## Model Status")
                status_box = gr.Markdown(model_status())

                refresh_button = gr.Button("Refresh Status")
                refresh_button.click(
                    fn=model_status,
                    inputs=None,
                    outputs=status_box,
                )

            with gr.Column(scale=2):
                gr.Markdown("## Upload Geometry")
                vtk_file = gr.File(
                    label="VTK file",
                    file_types=[".vtk"],
                    type="filepath",
                )

                gr.Markdown("## Flow Parameters")
                geom_name = gr.Textbox(label="geom_name", value="geom_976")
                alt_kft = gr.Textbox(label="alt_kft", value="2")
                reynolds = gr.Textbox(label="Re", value="17000000")
                mach = gr.Textbox(label="M_inf", value="0.28")
                alpha_deg = gr.Textbox(label="alpha_deg", value="5")
                beta_deg = gr.Textbox(label="beta_deg", value="0")

                simulate_button = gr.Button("Simulate", variant="primary")

        gr.Markdown("## Progress")
        status_text = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2,
        )

        gr.Markdown("## Prediction Summary")
        summary = gr.Markdown()

        gr.Markdown("## Download predicted VTK")
        output_file = gr.File(label="Predicted VTK", interactive=False)
        report_file = gr.File(label="Simulation Report", interactive=False)

        simulate_button.click(
            fn=simulate,
            inputs=[
                vtk_file,
                geom_name,
                alt_kft,
                reynolds,
                mach,
                alpha_deg,
                beta_deg,
            ],
            outputs=[
                status_text,
                summary,
                output_file,
                report_file,
                status_box,
            ],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
