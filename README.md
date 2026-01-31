---

# 🧠 GNN-based CFD Inference Pipeline

This repository contains an **end-to-end pipeline** for preprocessing CFD VTK data, training a **Graph Neural Network (GNN)** model, and running inference to generate predicted Cp fields and evaluation metrics.

The codebase has been refactored to be **configuration-driven**, modular, and easy to run selectively (preprocess / train / inference).

---

## 📌 Key Changes & Design Decisions (So Far)

### ✅ 1. Unified Configuration (`config.py`)

All tunable parameters are centralized in **`config.py`**, including:

* Paths (data, outputs, checkpoints)
* Model architecture
* Training hyperparameters
* Inference options
* Device selection (CPU / CUDA)

This removes hard-coded values from scripts and ensures **single-source-of-truth** configuration.

---

### ✅ 2. Modular Pipeline Structure

The pipeline is split into three explicit stages:

| Stage         | Script             | Purpose                          |
| ------------- | ------------------ | -------------------------------- |
| Preprocessing | `preprocessing.py` | Convert VTK → PyTorch graph data |
| Training      | `training.py`      | Train GNN model                  |
| Inference     | `inference.py`     | Predict Cp + save VTK + metrics  |

Each stage can be run **independently** or as part of the full pipeline.

---

### ✅ 3. Controlled Execution via `start.py`

The `start.py` script acts as the **entry point**.

Users can **comment / uncomment** stages depending on what they want to run:

```python
def main():
    print_config(config)

    print("Starting training pipeline...\n")

    pipeline_start = time.time()

    preprocessing.main()
    training.main()
    inference.main()

    time_ = time.time() - pipeline_start
    print(f"script completed , time taken : {time_:.2f} sec")


if __name__ == "__main__":
    main()
```

👉 This avoids multiple CLI commands and keeps execution simple.

---

### ✅ 4. Robust `case_data.dat` Handling

* Header-aware parsing
* Explicit column access (no positional indexing)
* Geometry name correctly mapped to `geom_params.ini`
* Prevents silent mismatches during inference

---

### ✅ 5. Consistent Geometry Parameter Handling

* `geom_params.ini` used consistently across:

  * preprocessing
  * training
  * inference
* Geometry parameters are broadcasted per-node and appended to node features

---

### ✅ 6. Dependency Setup Script (`install.sh`)

All required dependencies are installed via:

```bash
bash install.sh
```

This ensures environment reproducibility.

---

## 📁 Repository Structure

```
.
├── config.py                 # Central configuration
├── start.py                  # Pipeline launcher
├── install.sh                # Dependency installer
│
├── preprocessing.py          # VTK → Graph preprocessing
├── training.py               # Model training
├── inference.py              # Model inference + metrics
│
├── model.py                  # GNN architecture
├── geom_params.ini           # Geometry parameters
├── case_data.dat             # Case metadata (with headers)
│
├── data/                     # Raw VTK files
├── test/                     # Test VTKs for inference
├── ptfiles/                  # Preprocessed graph data
├── output/                   # Model checkpoints
└── predicted/                # Inference output VTKs
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone <repo_url>
cd <repo_name>
```

### 2️⃣ Install dependencies

```bash
bash install.sh
```

---

## 🚀 Running the Pipeline

### 🔹 Option A: Run Full Pipeline

Make sure all stages are **uncommented** in `start.py`:

```python
preprocessing.main()
training.main()
inference.main()
```

Then run:

```bash
python start.py
```

---

### 🔹 Option B: Run Only Specific Stages

Edit `start.py` and comment out unwanted steps.

#### Example: Inference only

```python
# preprocessing.main()
# training.main()
inference.main()
```

Then:

```bash
python start.py
```

---

## 🧪 Outputs

### ✔ Preprocessing

* `.pt` graph files saved to:

  ```
  ptfiles/
  ```

### ✔ Training

* Model checkpoints saved to:

  ```
  output/
  ```

### ✔ Inference

* Predicted VTK files saved to:

  ```
  predicted/
  ```
* Includes:

  * Predicted Cp
  * Error fields
  * Optional inference logs

---

## 🛠 Debugging Tips

* Verify `geom_name` exists in `geom_params.ini`
* Ensure `case_data.dat` headers are intact
* Confirm VTKs exist for all listed cases
* Print `case_df.head()` if inference skips cases

---

## 📌 Notes / Next Improvements

* CLI argument support (instead of commenting code)
* Shared data loader utility
* Automatic validation of case ↔ geometry consistency
* Multi-GPU / mixed precision training support

---

## ✨ Summary

This repo now provides:

* ✅ Clean, configurable pipeline
* ✅ Reproducible runs
* ✅ Modular execution
* ✅ Robust data handling