import time
import psutil
import os
import tempfile

def profile_model(model, sample_input):
    import numpy as np
    import tensorflow as tf

    start_time = time.time()
    model(sample_input)
    latency = time.time() - start_time

    param_count = model.count_params()
    size_params_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per float32
    memory = psutil.Process().memory_info().rss / (1024 * 1024)

    return {
        "latency (s)": latency,
        #"size (M params)": param_count / 1e6,
        "size (MB)": size_params_mb,
        "memory (MB)": memory
    }

def profile_tflite_model(tflite_model, sample_input):
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sample_input = sample_input.astype(np.float32)
    sample_input = np.expand_dims(sample_input, axis=0)
    interpreter.set_tensor(input_details[0]['index'], sample_input)

    start_time = time.time()
    interpreter.invoke()
    latency = time.time() - start_time

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(tflite_model)
        tmp_file_path = tmp_file.name

    model_size = os.path.getsize(tmp_file_path) / (1024 * 1024)  # in MB

    try:
        os.remove(tmp_file_path)
    except PermissionError:
        pass

    memory = psutil.Process().memory_info().rss / (1024 * 1024)

    return {
        "latency (s)": latency,
        "size (MB)": model_size,
        "memory (MB)": memory
    }
