# -----------------------------------------
# logger/result_logger.py
# -----------------------------------------

import pandas as pd
import os
from config import RESULTS_FILE

def save_results(results_dict):
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    df.index.name = "Model"
    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE, index_col=0)
        df = pd.concat([existing, df])
    df.to_csv(RESULTS_FILE)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(df)