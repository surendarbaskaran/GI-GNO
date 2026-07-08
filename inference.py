"""Single-geometry inference engine for GI-GNO."""

from __future__ import annotations

import configparser
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from config import (
    CHECKPOINT,
    DEVICE,
    DROPOUT,
    GEOM_PARAM_FILE,
    GRID_SIZE,
    HIDDEN_DIM,
    NODE_IN_DIM,
    NORM_STATS_FILE,
    NUM_GNN_LAYERS,
    OUT_DIM,
)
from model import GAGNO

OUTPUT_DIR = "/tmp/output"
FLOW_KEYS = ("alt_kft", "Re", "M_inf", "alpha_deg", "beta_deg")
GEOM_KEYS = ("B1", "B2", "B3", "C1", "C2", "C3", "C4", "S1", "S2", "S3")

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@dataclass
class InferenceState:
    """Cached model and static inference assets."""

    model: GAGNO
    device: torch.device
    x_mean: torch.Tensor
    x_std: torch.Tensor
    cp_mean: torch.Tensor
    cp_std: torch.Tensor
    geom_config: configparser.ConfigParser


@dataclass
class PreparedMesh:
    """Mesh tensors required by the unchanged inference pipeline."""

    mesh: pv.DataSet
    vertices: torch.Tensor
    edge_index: torch.Tensor
    x_fp32: torch.Tensor
    cp_true: torch.Tensor | None


_STATE: InferenceState | None = None


def load_model() -> InferenceState:
    """Load checkpoint, normalization stats, geometry config, and device once."""

    global _STATE
    if _STATE is not None:
        return _STATE

    _validate_asset_paths()
    device = torch.device(DEVICE)
    LOGGER.info("Loading GI-GNO model on %s", device)

    model = GAGNO(
        node_in_dim=NODE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        grid_size=GRID_SIZE,
        dropout=DROPOUT,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        LOGGER.warning(
            "Missing keys while loading checkpoint: %s",
            incompatible.missing_keys[:10],
        )
    if incompatible.unexpected_keys:
        LOGGER.warning(
            "Unexpected keys while loading checkpoint: %s",
            incompatible.unexpected_keys[:10],
        )
    model.eval()

    norm_stats = torch.load(NORM_STATS_FILE, map_location="cpu")
    geom_config = configparser.ConfigParser()
    geom_config.read(GEOM_PARAM_FILE)

    _STATE = InferenceState(
        model=model,
        device=device,
        x_mean=norm_stats["x_mean"].float(),
        x_std=norm_stats["x_std"].float(),
        cp_mean=norm_stats["cp_mean"].float(),
        cp_std=norm_stats["cp_std"].float(),
        geom_config=geom_config,
    )
    LOGGER.info("Inference assets loaded")
    return _STATE


def run_inference(vtk_path: str, input_json: dict[str, Any]) -> dict[str, Any]:
    """Run prediction for one uploaded VTK and return structured results."""

    start_time = time.time()
    try:
        state = load_model()
        inputs = _validate_request(vtk_path, input_json, state.geom_config)
        prepared = _prepare_mesh(vtk_path, inputs, state.geom_config)

        x = normalize(prepared.x_fp32, state.x_mean, state.x_std)
        data = Data(
            x=x,
            edge_index=prepared.edge_index,
            pos=prepared.vertices[:, :2],
        ).to(state.device)

        with torch.no_grad():
            cp_pred_norm = state.model(data).cpu().squeeze()
            cp_pred = cp_pred_norm * state.cp_std + state.cp_mean

        metrics = _compute_metrics(cp_pred, prepared.cp_true)
        cl_pred, cd_pred = compute_force_coefficients(
            prepared.mesh,
            cp_pred,
            inputs["alpha_deg"],
        )
        output_vtk = _save_prediction_vtk(
            vtk_path,
            prepared.mesh,
            cp_pred,
            prepared.cp_true,
        )
        elapsed = time.time() - start_time
        report_path = _write_report(
            output_vtk=output_vtk,
            vtk_path=vtk_path,
            input_json=inputs,
            metrics=metrics,
            forces={"CL": cl_pred, "CD": cd_pred},
            elapsed=elapsed,
        )

        return {
            "success": True,
            "output_vtk": output_vtk,
            "report": report_path,
            "vertices": int(prepared.mesh.n_points),
            "faces": int(prepared.mesh.n_cells),
            "inference_time": elapsed,
            "metrics": metrics,
            "forces": {
                "CL": cl_pred,
                "CD": cd_pred,
            },
        }
    except Exception as exc:  # noqa: BLE001 - UI/API should receive readable errors.
        LOGGER.exception("Inference failed")
        return {
            "success": False,
            "error": str(exc),
            "output_vtk": None,
            "report": None,
            "vertices": None,
            "faces": None,
            "inference_time": None,
            "metrics": _empty_metrics(),
            "forces": {
                "CL": None,
                "CD": None,
            },
        }


def compute_force_coefficients(
    mesh: pv.DataSet,
    cp_values: torch.Tensor,
    alpha_deg: float,
) -> tuple[float, float]:
    """Compute CL and CD from the predicted Cp distribution."""

    alpha = np.deg2rad(alpha_deg)
    surf = mesh.extract_surface().triangulate()
    surf = surf.compute_normals(cell_normals=True, point_normals=False)
    surf = surf.compute_cell_sizes()

    areas = np.array(surf.cell_data["Area"])
    normals = np.array(surf.cell_data["Normals"])
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    cp_point = cp_values.detach().cpu().numpy()
    cp_cell = cp_point[faces].mean(axis=1)

    fx_total = (-cp_cell * normals[:, 0] * areas).sum()
    fy_total = (-cp_cell * normals[:, 1] * areas).sum()

    cd = fx_total * np.cos(alpha) + fy_total * np.sin(alpha)
    cl = -fx_total * np.sin(alpha) + fy_total * np.cos(alpha)
    return float(cl), float(cd)


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply the training-time feature normalization."""

    return (x - mean) / std


def _validate_asset_paths() -> None:
    print(CHECKPOINT)
    print(NORM_STATS_FILE)
    print(GEOM_PARAM_FILE)
    required = {
        "checkpoint": CHECKPOINT,
        "normalization stats": NORM_STATS_FILE,
        "geometry file": GEOM_PARAM_FILE,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing model asset(s): " + "; ".join(missing))


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and any(
        key.startswith("encoder") or key.startswith("decoder")
        for key in checkpoint
    ):
        return checkpoint
    return checkpoint


def _validate_request(
    vtk_path: str,
    input_json: dict[str, Any],
    geom_config: configparser.ConfigParser,
) -> dict[str, Any]:
    if not vtk_path:
        raise ValueError("Uploaded file path is required.")

    path = Path(vtk_path)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded VTK file does not exist: {vtk_path}")
    if path.suffix.lower() != ".vtk":
        raise ValueError("Uploaded geometry must be a .vtk file.")

    if not isinstance(input_json, dict):
        raise ValueError("Flow parameters must be provided as a JSON object.")

    required_keys = ("geom_name", *FLOW_KEYS)
    missing = [key for key in required_keys if key not in input_json]
    if missing:
        raise ValueError("Missing required JSON key(s): " + ", ".join(missing))

    geom_name = str(input_json["geom_name"])
    if geom_name not in geom_config.sections():
        raise ValueError(f"geom_name '{geom_name}' was not found in geom.ini.")

    validated = {"geom_name": geom_name}
    for key in FLOW_KEYS:
        try:
            validated[key] = float(input_json[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"JSON key '{key}' must be numeric.") from exc
    return validated


def _prepare_mesh(
    vtk_path: str,
    input_json: dict[str, Any],
    geom_config: configparser.ConfigParser,
) -> PreparedMesh:
    mesh = pv.read(vtk_path)
    mesh = mesh.connectivity(extraction_mode="largest")

    vertices_np = mesh.points.astype(np.float32)
    center = vertices_np.mean(axis=0)
    vertices_np -= center
    scale = np.max(np.abs(vertices_np))
    vertices_np /= scale + 1e-8
    vertices = torch.tensor(vertices_np, dtype=torch.float32)

    faces = torch.tensor(mesh.faces.reshape(-1, 4)[:, 1:], dtype=torch.long)
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        dim=0,
    )
    edge_index = to_undirected(edges.T)

    geom_name = input_json["geom_name"]
    geom_params = torch.tensor(
        [geom_config.getfloat(geom_name, key) for key in GEOM_KEYS],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(vertices.size(0), 1)

    flow_params = torch.tensor(
        [float(input_json[key]) for key in FLOW_KEYS],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(vertices.size(0), 1)

    x_fp32 = torch.cat([vertices, flow_params, geom_params], dim=1)
    cp_true = None
    if "cp" in mesh.point_data:
        cp_true = torch.tensor(mesh.point_data["cp"], dtype=torch.float32)

    return PreparedMesh(
        mesh=mesh,
        vertices=vertices,
        edge_index=edge_index,
        x_fp32=x_fp32,
        cp_true=cp_true,
    )


def _compute_metrics(
    cp_pred: torch.Tensor,
    cp_true: torch.Tensor | None,
) -> dict[str, float | None]:
    if cp_true is None:
        return _empty_metrics()

    diff = cp_pred - cp_true
    mse = torch.mean(diff ** 2).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    mae = torch.mean(torch.abs(diff)).item()
    rel_l2 = (torch.norm(diff) / (torch.norm(cp_true) + 1e-8)).item()
    max_abs_error = torch.max(torch.abs(diff)).item()

    corr = np.corrcoef(
        cp_true.numpy().flatten(),
        cp_pred.numpy().flatten(),
    )[0, 1]

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mse": float(mse),
        "rel_l2": float(rel_l2),
        "corr": float(corr),
        "max_abs_error": float(max_abs_error),
    }


def _empty_metrics() -> dict[str, None]:
    return {
        "rmse": None,
        "mae": None,
        "mse": None,
        "rel_l2": None,
        "corr": None,
        "max_abs_error": None,
    }


def _save_prediction_vtk(
    vtk_path: str,
    mesh: pv.DataSet,
    cp_pred: torch.Tensor,
    cp_true: torch.Tensor | None,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / f"predicted_{Path(vtk_path).name}"

    out_mesh = mesh.copy()
    out_mesh.point_data["Cp_pred"] = cp_pred.detach().cpu().numpy()
    if cp_true is not None:
        diff = cp_pred - cp_true
        out_mesh.point_data["Cp_error"] = diff.detach().cpu().numpy()

    out_mesh.save(output_path)
    return str(output_path)


def _write_report(
    output_vtk: str,
    vtk_path: str,
    input_json: dict[str, Any],
    metrics: dict[str, float | None],
    forces: dict[str, float],
    elapsed: float,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = Path(OUTPUT_DIR) / "simulation_report.txt"
    timestamp = datetime.now().isoformat(timespec="seconds")

    lines = [
        "GI-GNO CFD Simulation Report",
        "=" * 30,
        f"Geometry: {Path(vtk_path).name}",
        f"Predicted VTK: {output_vtk}",
        f"Execution time: {elapsed:.6f} seconds",
        f"Timestamp: {timestamp}",
        "",
        "Flow conditions",
        "-" * 15,
        f"geom_name: {input_json['geom_name']}",
        f"alt_kft: {input_json['alt_kft']}",
        f"Re: {input_json['Re']}",
        f"M_inf: {input_json['M_inf']}",
        f"alpha_deg: {input_json['alpha_deg']}",
        f"beta_deg: {input_json['beta_deg']}",
        "",
        "Metrics",
        "-" * 7,
        f"RMSE: {_format_value(metrics['rmse'])}",
        f"MAE: {_format_value(metrics['mae'])}",
        f"MSE: {_format_value(metrics['mse'])}",
        f"Relative L2: {_format_value(metrics['rel_l2'])}",
        f"Correlation: {_format_value(metrics['corr'])}",
        f"Maximum Error: {_format_value(metrics['max_abs_error'])}",
        "",
        "Force coefficients",
        "-" * 18,
        f"CL: {_format_value(forces['CL'])}",
        f"CD: {_format_value(forces['CD'])}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def _format_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6e}"


def main() -> None:
    """Preserve script entry point with a clear interactive-app hint."""

    LOGGER.info("Use inference_gradio.py or call run_inference(vtk_path, input_json).")


if __name__ == "__main__":
    main()
