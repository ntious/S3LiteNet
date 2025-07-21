import pandas as pd
import os
from glob import glob
# Path to the folder containing the CSV files
folder_path = "results"
# List all CSV files in the folder
csv_files = glob(os.path.join(folder_path, "*.csv"))

# Container for all DataFrames
df_list = []

# Loop through each file
for file in csv_files:
    df = pd.read_csv(file, header=0)
    
    # Rename the first unnamed column to "Model"
    if df.columns[0].startswith("Unnamed") or df.columns[0] == '':
        df.rename(columns={df.columns[0]: "Model"}, inplace=True)
    
    # Add Dataset column
    if "ton_iot_modbus" in file.lower():
        df["Dataset"] = "TON_IOT_Modbus"
    elif "cic_iot_diad" in file.lower():
        df["Dataset"] = "CIC_IOT_DIAD"
    elif "ton_iot_thermostat" in file.lower():
        df["Dataset"] = "TON_IoT_Thermostat"
    else:
        df["Dataset"] = "Unknown"

    # Add DL algorithm column
    filename = os.path.basename(file).lower()
    if "cnn_gru_comparison" in filename:
        df["DL algorithm"] = "CNN & GRU"
    elif "cnn_comparison" in filename:
        df["DL algorithm"] = "CNN"
    elif "lstm_comparison" in filename:
        df["DL algorithm"] = "LSTM"
    elif "gru_comparison" in filename:
        df["DL algorithm"] = "GRU"
    elif "simplernn_comparison" in filename:
        df["DL algorithm"] = "Simple RNN"
    elif "transformer_comparison" in filename:
        df["DL algorithm"] = "Distilled Transformer"
    else:
        df["DL algorithm"] = "Unknown"

    # Append to the list
    df_list.append(df)
print(f"Found {len(csv_files)} CSV files in {folder_path}")
print(csv_files)
# Combine all DataFrames
df_combined = pd.concat(df_list, ignore_index=True)

# Reorder columns
ordered_cols = ["Model", "Dataset", "DL algorithm", "accuracy", "precision", "recall", "f1", "auc",
                "latency (s)", "size (MB)", "memory (MB)", "params"]
df_combined = df_combined[ordered_cols]

# Optional: save the combined DataFrame
df_combined.to_csv("analysis/combined_results.csv", index=False)

# Display the result
print(df_combined.head())
