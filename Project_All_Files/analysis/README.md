# Create a README.md summarizing the analyses performed
readme_content = """
# Analysis of Combined Lightweight Anomaly Detection Results

This document summarizes the statistical and visual analyses conducted on the benchmarking results of lightweight deep learning models for edge-based anomaly detection. The analyses are structured around the study's four research questions and the contributions made to the field.

---

## 🔍 Research Questions and Corresponding Analyses

### **RQ1**: *How effectively can widely-used lightweight neural network models detect anomalies in standard datasets, as measured by accuracy, precision, recall, and F1-score?*

**Analysis Performed:**
- **One-Way ANOVA**: Tested if detection metrics (accuracy, precision, recall, F1, AUC) differ significantly across DL algorithms.
- **Tukey HSD Post-hoc Test**: Identified which pairs of DL models showed statistically significant differences in F1-score.
- **Visualization**: Boxplots of F1 scores across DL algorithms.

**Purpose:** Validates that certain DL algorithms consistently outperform others, strengthening model performance claims.

---

### **RQ2**: *How do the inference latency, memory footprint, and model size of the chosen models compare under hardware limitations?*

**Analysis Performed:**
- **One-Way ANOVA**: Compared latency, memory, model size, and parameter count across DL algorithms.

**Purpose:** Confirms that different models present statistically meaningful differences in deployment efficiency, justifying resource-related trade-offs.

---

### **RQ3**: *What trade-offs exist between model accuracy and deployment efficiency across datasets and architectures?*

**Analysis Performed:**
- **Spearman Correlation**: Measured monotonic relationships between F1 score and efficiency metrics (latency, memory, size).
- **Visualization**: Scatter plots of F1 vs. latency, memory, and model size.

**Purpose:** Quantifies trade-offs to inform real-world deployment choices under resource constraints.

---

### **RQ4**: *Which model(s) exhibit the best balance of performance and efficiency for real-time edge deployment?*

**Analysis Performed:**
- **Composite Score Calculation**: Combined normalized performance and inverse-efficiency metrics to rank models.
- **Top 10 Models Identified**: Exported ranked models to CSV.
- **Visualization**: Bar plot of top combined scores by DL algorithm.

**Purpose:** Empirically supports claims on ideal model selection for practical edge-based applications.

---

### **RQ5 (Bonus)**: *Does model performance vary significantly depending on the dataset type?*

**Analysis Performed:**
- **Two-Way ANOVA**: Assessed main and interaction effects between DL algorithm and Dataset on F1 score.

**Purpose:** Shows that both dataset type and model architecture significantly influence performance, affirming the need for robust cross-dataset evaluation.

---

## 📁 Output Structure

- `analysis/image/`: High-quality plots (.png) for visual interpretation of results
- `analysis/table/`: CSV files of statistical summaries and test results
  - `rq1_anova_results.csv`
  - `rq2_anova_efficiency.csv`
  - `rq3_correlation_f1_vs_efficiency.csv`
  - `rq4_top_models.csv`
  - `rq5_interaction_anova.csv`
  - `rq1_tukey_f1.txt` (post-hoc comparison)

---

## 📘 Notes

- All statistical tests used a 95% confidence level (alpha = 0.05).
- When assumptions of normality may not hold, nonparametric alternatives like Spearman correlation were applied.

This README is designed to help you or future collaborators understand the reasoning behind each statistical method and how it connects to the overarching research goals.

"""
