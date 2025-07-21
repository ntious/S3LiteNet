# 🧠 S3LiteNet: Lightweight Sparse Neural Models for Real-Time Edge-Based Anomaly Detection

**S3LiteNet** is a multi-phase research and development initiative aimed at advancing lightweight neural network models for anomaly detection at the edge, particularly in Information-Centric Networking (ICN) and IoT contexts. This repository documents the full pipeline — from systematic literature review to benchmarking, and ultimately the design of a custom sparse model optimized for edge deployment.

---

## 🔍 Project Overview

| Phase | Title | Status | Focus |
|-------|-------|--------|-------|
| **1** | Systematic Literature Review | 90% Complete | Analyze trends, gaps, and deployment barriers in existing edge-based lightweight models |
| **2** | Benchmarking Study | Complete | Compare CNN, LSTM, GRU, Transformer, and hybrids across accuracy, latency, memory, and model size |
| **3** | S3LiteNet Development | In Progress | Design a custom sparse neural network optimized for edge inference with explainability |

---

## 📁 Repository Structure

```
S3LiteNet/
│
├── Phase_1_Literature_Review/     # PRISMA protocol, extracted data, and analysis scripts
├── Phase_2_Comparative_Study/       # Source code for training, evaluating, and profiling models
├── Phase_3_S3LiteNet_Model/       # Work-in-progress implementation of the S3LiteNet architecture
├── paper/                         # IEEE manuscript, pre-print, and related figures
├── results/                       # Aggregated benchmarking results and visualizations
└── README.md                      # You're here!
```

---

## 📌 Research Goals

- ⚡ **Efficiency**: Reduce computational and memory overhead for real-time inference.
- 🔍 **Explainability**: Enhance transparency through integrated XAI techniques (e.g., SHAP, attention).
- 🛰️ **Deployment**: Support deployment in ICN and IoT environments with limited connectivity.
- 🧠 **Adaptability**: Address concept drift, data imbalance, and edge-device heterogeneity.

---

## 📝 How to Cite

If you use this repository, please cite our preprint:

> Nti et al. “Evaluating Lightweight Neural Models for Edge-Based Anomaly Detection: Performance and Efficiency Trade-offs.” *Research Square Preprint* (2025). [https://doi.org/10.21203/rs.3.rs-7138288/v1](https://doi.org/10.21203/rs.3.rs-7138288/v1)

**BibTeX:**
```bibtex
@article{asiedu2025s3litenet,
  title={Evaluating Lightweight Neural Models for Edge-Based Anomaly Detection: Performance and Efficiency Trade-offs},
  author={Nti and Others},
  journal={Research Square},
  year={2025},
  doi={10.21203/rs.3.rs-7138288/v1}
}
```

---

## 🧪 Publications and Manuscripts

- 📄 IEEE Manuscript (Under Review): `paper/IEEE Manuscript_draft_R2.pdf`
- 📄 Preprint (Published): [`paper_pre_print.pdf`](https://assets-eu.researchsquare.com/files/rs-7138288/v1_covered_dcf2375f-29da-45bd-b2a9-6d4f1f76dc9d.pdf?c=1752753911)
