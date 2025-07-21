import torch
import torch.nn as nn
import torch_pruning as tp
import torch.quantization
import pandas as pd
import time
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# ========================
# Load and preprocess
# ========================
df = pd.read_csv("data/Combined_CIC_IoT_DIAD_2024.csv")
df.drop(columns=["Src IP", "Dst IP", "Flow ID", "Timestamp", "Attack Name"], inplace=True, errors='ignore')
X = df.drop(columns=["Label"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
input_dim = X_test_tensor.shape[1]
example_input = torch.randn(1, input_dim)

# ========================
# Model Definitions
# ========================

class BaselineCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv(x))).squeeze(-1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)

class OptimizedCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.bn(self.conv(x)))).squeeze(-1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)

class QuantizedOptimizedCNN(OptimizedCNN):
    def __init__(self, input_dim):
        super().__init__(input_dim)

# ========================
# Pruning
# ========================
def prune_model(model, example_input, amount=0.5):
    DG = tp.DependencyGraph().build_dependency(model, example_input)
    strategy = tp.strategy.L1Strategy()
    for layer in [m for m in model.modules() if isinstance(m, nn.Conv1d)]:
        idxs = strategy(layer.weight, amount=amount)
        plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=idxs)
        plan.exec()
    return model

# ========================
# Quantization
# ========================
def quantize_model(model, example_input):
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    model(example_input)  # Calibration
    torch.quantization.convert(model, inplace=True)
    return model

# ========================
# Evaluation
# ========================
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
    size = sum(p.numel() for p in model.parameters()) / 1e6
    memory = psutil.Process().memory_info().rss / 1e6
    return {"latency": latency, "size (M params)": size, "memory (MB)": memory}

# ========================
# Run Benchmark
# ========================
baseline = prune_model(BaselineCNN(input_dim), example_input, amount=0.3)
optimized = prune_model(OptimizedCNN(input_dim), example_input, amount=0.5)
quantized = prune_model(QuantizedOptimizedCNN(input_dim), example_input, amount=0.5)
quantized = quantize_model(quantized, example_input)

results = {}
for name, model in zip(
    ["Baseline CNN", "Optimized CNN", "Quantized Optimized CNN"],
    [baseline, optimized, quantized]
):
    model.eval()
    preds = model(X_test_tensor).detach().numpy().round()
    results[name] = {**compute_metrics(y_test_tensor, preds), **profile_model(model, X_test_tensor[:1])}

# ========================
# Save Results
# ========================
df_results = pd.DataFrame(results).T
df_results.to_csv("cnn_comparison_results.csv")
print(df_results)
