import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Set up output directories
os.makedirs("images", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# Load the combined results file
df_combined = pd.read_csv("combined_results.csv")

# Define metric categories
performance_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
efficiency_metrics = ['latency (s)', 'size (MB)', 'memory (MB)', 'params']

# === RQ1: ANOVA on performance metrics ===
anova_results = []
for metric in performance_metrics:
    groups = [df_combined[df_combined["DL algorithm"] == algo][metric].dropna() 
              for algo in df_combined["DL algorithm"].unique()]
    if all(len(group) > 1 for group in groups):
        f_stat, p_val = f_oneway(*groups)
        anova_results.append({"Metric": metric, "Test": "ANOVA", "F-Statistic": f_stat, "p-value": p_val})
pd.DataFrame(anova_results).to_csv("tables/rq1_anova_results.csv", index=False)

# Tukey HSD on F1
tukey_result = pairwise_tukeyhsd(endog=df_combined["f1"], 
                                 groups=df_combined["DL algorithm"], alpha=0.05)
with open("tables/rq1_tukey_f1.txt", "w") as f:
    f.write(str(tukey_result))

# F1 Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_combined, x="DL algorithm", y="f1")
plt.title("F1 Score by DL Algorithm")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("images/rq1_f1_boxplot.png")
plt.close()

# === RQ2: ANOVA on deployment efficiency ===
anova_eff_results = []
for metric in efficiency_metrics:
    groups = [df_combined[df_combined["DL algorithm"] == algo][metric].dropna() 
              for algo in df_combined["DL algorithm"].unique()]
    if all(len(group) > 1 for group in groups):
        f_stat, p_val = f_oneway(*groups)
        anova_eff_results.append({"Metric": metric, "Test": "ANOVA", "F-Statistic": f_stat, "p-value": p_val})
pd.DataFrame(anova_eff_results).to_csv("tables/rq2_anova_efficiency.csv", index=False)

# === RQ3: Spearman correlation between F1 and efficiency ===
correlation_results = []
for metric in efficiency_metrics:
    corr, p_val = spearmanr(df_combined["f1"], df_combined[metric])
    correlation_results.append({
        "X": metric, "Y": "f1", "Method": "Spearman",
        "Correlation Coefficient": corr, "p-value": p_val
    })
pd.DataFrame(correlation_results).to_csv("tables/rq3_correlation_f1_vs_efficiency.csv", index=False)

# === RQ5: Two-way ANOVA interaction ===
# Rename column to avoid formula issues
df_ = df_combined.rename(columns={"DL algorithm": "DL_algorithm"})
model = ols('f1 ~ C(DL_algorithm) + C(Dataset) + C(DL_algorithm):C(Dataset)', data=df_).fit()
anova_interaction = sm.stats.anova_lm(model, typ=2)
anova_interaction.to_csv("tables/rq5_interaction_anova.csv")
print("I am done")
