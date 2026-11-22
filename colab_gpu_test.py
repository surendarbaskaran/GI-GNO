"""Quick check that a Colab runtime is up and has a GPU (e.g. a T4)."""
import platform
import subprocess

print("Python:", platform.python_version())
print("Host:", platform.node())

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("GPU:", out)
except (FileNotFoundError, subprocess.CalledProcessError):
    print("GPU: none found (set Runtime > Change runtime type > T4 GPU)")

try:
    import torch
    print("torch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
except ImportError:
    print("torch not installed")
