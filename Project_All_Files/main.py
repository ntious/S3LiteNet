import argparse
import importlib.util
import os
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = (PROJECT_ROOT / "models").resolve()

MODEL_SCRIPTS = {
    "cnn": "cnn_pruned.py",
    "lstm": "lstm_quantized.py",
    "transformer": "transformer_tiny.py",
    "gru": "gru_pruned.py",
    "cnn_gru": "cnn_gru_pruned.py",
    "simplernn": "simplernn_pruned.py",
}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def resolve_model_path(model_key):
    """Return an allowlisted model implementation inside the trusted model directory."""
    if model_key not in MODEL_SCRIPTS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_SCRIPTS.keys())}")
    path = (MODEL_DIR / MODEL_SCRIPTS[model_key]).resolve(strict=True)
    if path.parent != MODEL_DIR:
        raise ValueError("Resolved model path escaped the trusted model directory.")
    return path


def run_model(model_key):
    try:
        path = resolve_model_path(model_key)
    except (ValueError, FileNotFoundError) as exc:
        print(exc)
        return
    print(f"Running {model_key.upper()} model from {path}")
    spec = importlib.util.spec_from_file_location(model_key, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trusted model module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run selected model benchmark.")
    parser.add_argument("--model", type=str, required=True, help="cnn | lstm | transformer | gru | cnn_gru | simplernn")
    args = parser.parse_args()
    run_model(args.model)
