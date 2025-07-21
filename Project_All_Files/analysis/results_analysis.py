import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
# Create output directories
os.makedirs("images", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# Set seaborn style
sns.set(style="whitegrid", font_scale=1.2)
df_combined = pd.read_csv("combined_results.csv")
# Save summary stats table
df_combined.describe().to_csv("tables/summary_statistics.csv")

# --- RQ1: Detection Performance Across Models and Datasets ---
performance_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

# Melt for easier plotting
df_perf_melted = df_combined.melt(id_vars=["Model", "Dataset", "DL algorithm"], 
                                  value_vars=performance_metrics, 
                                  var_name="Metric", value_name="Score")

# Plot detection performance
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_perf_melted, x="Metric", y="Score", hue="DL algorithm")
plt.title("RQ1: Detection Performance by DL Algorithm")
plt.tight_layout()
plt.savefig("images/rq1_detection_performance.png")
plt.close()

# --- RQ2: Deployment Efficiency ---
efficiency_metrics = ['latency (s)', 'size (MB)', 'memory (MB)', 'params']
df_eff_melted = df_combined.melt(id_vars=["Model", "Dataset", "DL algorithm"], 
                                 value_vars=efficiency_metrics, 
                                 var_name="Metric", value_name="Value")

# Plot efficiency metrics
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_eff_melted, x="Metric", y="Value", hue="DL algorithm")
plt.title("RQ2: Deployment Efficiency by DL Algorithm")
plt.tight_layout()
plt.savefig("images/rq2_efficiency_metrics.png")
plt.close()

# --- RQ3: Trade-offs Between Accuracy and Efficiency ---
# Accuracy vs latency
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_combined, x="latency (s)", y="accuracy", hue="DL algorithm", style="Dataset")
plt.title("RQ3: Accuracy vs Latency")
plt.tight_layout()
plt.savefig("images/rq3_accuracy_vs_latency.png")
plt.close()

# Accuracy vs memory
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_combined, x="memory (MB)", y="accuracy", hue="DL algorithm", style="Dataset")
plt.title("RQ3: Accuracy vs Memory")
plt.tight_layout()
plt.savefig("images/rq3_accuracy_vs_memory.png")
plt.close()

# Accuracy vs model size
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_combined, x="size (MB)", y="accuracy", hue="DL algorithm", style="Dataset")
plt.title("RQ3: Accuracy vs Model Size")
plt.tight_layout()
plt.savefig("images/rq3_accuracy_vs_size.png")
plt.close()

# --- RQ4: Ideal Balance (Top performers by efficiency + performance) ---
# Normalize relevant metrics for multi-criteria scoring
df_normalized = df_combined.copy()
df_normalized["efficiency_score"] = (1 / (df_normalized["latency (s)"] + 1e-6)) + \
                                     (1 / (df_normalized["memory (MB)"] + 1e-6)) + \
                                     (1 / (df_normalized["size (MB)"] + 1e-6))

df_normalized["performance_score"] = df_normalized[["accuracy", "precision", "recall", "f1", "auc"]].mean(axis=1)

df_normalized["combined_score"] = df_normalized["performance_score"] * df_normalized["efficiency_score"]

top_models = df_normalized.sort_values("combined_score", ascending=False).head(10)
top_models.to_csv("tables/rq4_top_models.csv", index=False)

# Plot top models
plt.figure(figsize=(10, 6))
sns.barplot(data=top_models, x="combined_score", y="Model", hue="DL algorithm")
plt.title("RQ4: Top Models by Combined Efficiency and Performance")
plt.tight_layout()
plt.savefig("images/rq4_top_models.png")
plt.close()
print("I am done")