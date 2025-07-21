# Evaluating Lightweight Neural Models for Edge-Based Anomaly Detection: Performance and Efficiency Trade-offs
## Usage instructions and project overview

readme_path = Path("3SLiteNet/Project_All_Files/Readme.md")
readme_content = '''
# 🧠 TensorFlow Lightweight Model Benchmark
This project compares lightweight neural network architectures using TensorFlow, with support for pruning, quantization, and resource profiling. Each includes baseline, optimized, and quantized + pruned versions.
### Summary of Models:
* CNN
* LSTM
* GRU
* Transformer
* CNN + GRU hybrid
* SimpleRNN

## 📁 Project Structure

'''
benchmark_structured/
│
├── config.py # Shared hyperparameters and dataset path
├── requirements.txt # Python dependencies
├── main.py # CLI runner for model benchmarking
├── dashboard.py # Dashboard for viewing modules
│
├── data/
│ └── preprocess.py # SMOTE, normalization, train/test split
│
├── models/
│ ├── cnn_pruned.py # Baseline/Optimized/Quantized CNN
│ ├── lstm_quantized.py # LSTM model variants
│ └── transformer_tiny.py # Transformer variants
│ ├── cnn_gru_pruned.py # Baseline/Optimized/Quantized CNN_gru
│ ├── gru_prune.py # GRU model variants
│ └── simplernn_prunes.py # simple learn variants
├── train_eval/
│ ├── train.py # Unified training function
│ └── metrics.py # Evaluation metrics
│
├── benchmark/
│ └── resource_profiler.py # Latency, model size, memory usage
│
├── logger/
│ └── result_logger.py # Centralized result saving to CSV
│
└── results/ # CSV outputs for each model


---

##  How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt

### Step 2: Run a specific model
python main.py --model cnn
python main.py --model lstm
python main.py --model transformer
python main.py --model gru
python main.py --model cnn_gru
python main.py --model simplernn
#### To Run the Dashboard
streamlit run dashboard.py

## Features
✂️ Pruning with TensorFlow Model Optimization Toolkit
📦 Quantization using TFLite
🧪 Standard metrics: Accuracy, Precision, Recall, F1, AUC
📉 Resource profiling: latency, size, memory
📁 Centralized CSV logging to results/ folder

## Output
* Each model script saves a CSV file like:
* results/tensorflow_cnn_comparison.csv
* results/tensorflow_lstm_comparison.csv
* results/tensorflow_transformer_comparison.csv

## Extend This
You can add more models like GRUs, MLPs, or MobileNet variants by dropping new scripts into models/ and updating main.py.