
import argparse
import importlib.util
import os

MODEL_SCRIPTS = {
    "cnn": "models/cnn_pruned.py",
    "lstm": "models/lstm_quantized.py",
    "transformer": "models/transformer_tiny.py"
}

def run_model(model_key):
    if model_key not in MODEL_SCRIPTS:
        print(f"❌ Unknown model: {model_key}. Choose from {list(MODEL_SCRIPTS.keys())}")
        return
    path = MODEL_SCRIPTS[model_key]
    print(f"🚀 Running {model_key.upper()} model from {path}")
    spec = importlib.util.spec_from_file_location(model_key, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run selected model benchmark.")
    parser.add_argument("--model", type=str, required=True, help="cnn | lstm | transformer")
    args = parser.parse_args()
    run_model(args.model)
