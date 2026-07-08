# GI-GNO CFD Inference

This project contains the existing GI-GNO preprocessing, training, model, and
inference code. The inference path has been refactored into a reusable
single-geometry engine plus an interactive Gradio application.

## Project Files

| File | Purpose |
| --- | --- |
| `preprocessing.py` | Builds graph tensors and normalization statistics from VTK data. |
| `training.py` | Trains the GI-GNO model. |
| `model.py` | Defines the GNN/FNO model architecture. |
| `config.py` | Stores paths, model dimensions, and training settings. |
| `inference.py` | Loads the model once and runs inference on one uploaded VTK. |
| `inference_gradio.py` | Gradio web UI for interactive simulation. |

## Installation

Install the Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The project also requires PyTorch and PyTorch Geometric. Install versions that
match your CUDA or CPU environment from the official PyTorch and PyG install
instructions.

Required runtime packages include:

```text
torch
torch-geometric
pyvista
numpy
gradio
```

## Required Assets

Inference reads these paths from `config.py`:

```python
CHECKPOINT = "checkpoints/bestmodel/best_model.pt"
NORM_STATS_FILE = "normalization_stats.pt"
GEOM_PARAM_FILE = "dataset/geom_params.ini"
```

Place the trained checkpoint, normalization statistics, and geometry parameter
file at those configured paths before launching the app.

## Running The Gradio App

From the project root:

```bash
python inference_gradio.py
```

Open the local URL printed by Gradio. In Google Colab, launch with sharing if
needed:

```python
from inference_gradio import build_app

build_app().launch(share=True)
```

## Google Colab Setup

1. Upload or clone the project into Colab.
2. Install dependencies.
3. Upload the trained checkpoint to the path set by `CHECKPOINT`.
4. Upload `normalization_stats.pt` to the path set by `NORM_STATS_FILE`.
5. Upload `geom_params.ini` to the path set by `GEOM_PARAM_FILE`.
6. Run the Gradio launch cell.

The app writes predictions and reports to:

```text
/tmp/output
```

## How To Upload A VTK

Use the **Upload Geometry** section of the Gradio page and choose a `.vtk` file.
The uploaded mesh is processed with the same centering, scaling, graph
construction, feature ordering, normalization, prediction, and Cp
denormalization used by the original inference pipeline.

## How To Enter JSON Inputs

The reusable Python function accepts flow parameters as a dictionary:

```python
run_inference(
    vtk_path="example.vtk",
    input_json={
        "geom_name": "BWB_047",
        "alt_kft": 20,
        "Re": 17000000,
        "M_inf": 0.28,
        "alpha_deg": 5,
        "beta_deg": 0,
    },
)
```

In the Gradio app, enter the same values in the vertical flow-parameter
textboxes. The UI builds the dictionary internally.

## How To Run Simulation

1. Confirm **Model Status** shows the checkpoint, normalization statistics, and
   geometry config are available.
2. Upload a `.vtk` geometry.
3. Enter `geom_name`, `alt_kft`, `Re`, `M_inf`, `alpha_deg`, and `beta_deg`.
4. Click **Simulate**.
5. Watch the progress indicator and status message.

Validation checks that the file exists, the extension is `.vtk`, all required
flow keys are present, numeric values can be parsed, and `geom_name` exists in
`geom_params.ini`.

## Outputs

After inference, the app saves:

```text
/tmp/output/predicted_<filename>.vtk
/tmp/output/simulation_report.txt
```

The prediction summary displays:

```text
Vertices
Faces
Inference Time
RMSE
MAE
Correlation
CL
CD
```

If the uploaded VTK contains a `cp` point-data field, error metrics and
`Cp_error` are generated. If no true `cp` field exists, the app still saves
`Cp_pred`, and error metrics are shown as `N/A`.

## Downloading Results

Use the download controls at the bottom of the Gradio page to download the
predicted VTK and the simulation report.
