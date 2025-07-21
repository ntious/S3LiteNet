# -----------------------------------------
# logger/result_logger.py
# -----------------------------------------
import os
import pandas as pd
from config import DATASET_PATH

def get_data_name(filename_):
    """Dispatch to appropriate dataset-specific name based on path name."""
    path = DATASET_PATH.lower()

    if "cic_iot_diad" in path:
        return "cic_iot_diad_"+filename_
    elif "ton_iot" in path:
        return "ton_iot_"+filename_
    else:
        return filename_

def save_results(results_dict, filename):
    df = pd.DataFrame(results_dict).T
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", get_data_name(filename))
    df.to_csv(path)
    print(f"\nResults saved to {path}")
    print(df)
