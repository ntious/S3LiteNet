# test_models.py

import pandas as pd
import torch
import time, os, psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Local imports
from models.cnn_pruned import get_model as cnn_model
from models.lstm_quantized import get_model as lstm_model
from models.transformer_tiny import get_model as transformer_model

# Config
DATASET_PATH = r"C:\Users\ntiik\3SLiteNet\Comparative Study\data\Combined_CIC_IoT_DIAD_2024.csv" # update if needed
#3SLiteNet/Comparative Study/test_script.py
TARGET_COLUMN = "Label"
RESULTS_FILE = "benchmark_results.csv"
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Load and preprocess
df2 = pd.read_csv(DATASET_PATH)


df = df2.sample(n=1000, random_state=42)

df.drop(columns=["Src IP", "Dst IP", "Flow ID", "Timestamp", "Attack Name"], inplace=True, errors='ignore')

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

sm = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Init models
input_dim = X_train_tensor.shape[1]
models = {
    "Pruned CNN": cnn_model(input_dim),
    "Quantized LSTM": lstm_model(input_dim),
    "Distilled Transformer": transformer_model(input_dim)
}

# Eval functions
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred)
    }

def profile_model(model, sample_input):
    start_time = time.time()
    with torch.no_grad():
        _ = model(sample_input)
    latency = time.time() - start_time
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    memory = psutil.Process(os.getpid()).memory_info().rss / 1e6
    return {
        "latency": latency,
        "size (M params)": model_size,
        "memory (MB)": memory
    }

# Run benchmarks
results = {}
for name, model in models.items():
    print(f"\nRunning: {name}")
    model.eval()
    try:
        preds = model(X_test_tensor).detach().numpy().round()
        metrics = compute_metrics(y_test_tensor, preds)
        resource = profile_model(model, X_test_tensor[:1])
        results[name] = {**metrics, **resource}
    except Exception as e:
        print(f"Error: {e}")
        results[name] = {"error": str(e)}

# Save results
results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
results_df.to_csv(RESULTS_FILE)
print(f"\n✅ Results saved to {RESULTS_FILE}")
print(results_df)

