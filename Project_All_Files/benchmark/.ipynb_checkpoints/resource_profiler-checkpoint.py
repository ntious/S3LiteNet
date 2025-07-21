
import time
import psutil

def profile_model(model, sample_input):
    import numpy as np
    import tensorflow as tf
    start_time = time.time()
    model(sample_input)
    latency = time.time() - start_time
    size = model.count_params() / 1e6
    memory = psutil.Process().memory_info().rss / 1e6
    return {
        "latency": latency,
        "size (M params)": size,
        "memory (MB)": memory
    }
