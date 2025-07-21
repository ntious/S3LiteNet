# CNN + GRU Benchmarking Script
# Baseline, Optimized, and Quantized + Pruned model variants

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pandas as pd
import numpy as np

from config import RANDOM_STATE, EPOCHS, OPTIMIZER_VAL, LOSS_VAL,BATCH_SIZE  
from data.preprocess import load_and_preprocess
from train_eval.train import train_model
from train_eval.metrics import evaluate_model
from benchmark.resource_profiler import profile_model, profile_tflite_model
from logger.result_logger import save_results

# Load the preprocessed dataset
X_train, X_test, y_train, y_test = load_and_preprocess()

# --- Hybrid CNN + GRU Architectures ---

# Baseline: 1D CNN followed by GRU
def build_baseline_cnn_gru(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Optimized: smaller Conv1D, reduced GRU, pooling
def build_optimized_cnn_gru(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        tf.keras.layers.Conv1D(16, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GRU(16, return_sequences=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Prune model using polynomial sparsity schedule
def apply_pruning(model):
    schedule = tfmot.sparsity.keras.PolynomialDecay(0.0, 0.5, 0, 1000)
    return tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=schedule)

# Quantize with TensorFlow Lite converter
def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable fallback to TF ops (for GRU support)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # Prevent illegal lowering of TensorList ops
    converter._experimental_lower_tensor_list_ops = False

    # Enable post-training quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    return converter.convert()


# Evaluate TFLite model
def evaluate_tflite(tflite_model, X_test, y_test, param_count):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    for x in X_test.astype(np.float32):
        x = np.expand_dims(x, axis=0)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        preds.append(int(output[0][0] > 0.5))

    class TFLiteModelWrapper:
        def __init__(self, predictions):
            self.preds = predictions
        def predict(self, X):
            return np.array(self.preds).reshape(-1, 1)

    return evaluate_model(TFLiteModelWrapper(preds), X_test, y_test) | {"params": param_count}

# --- Training & Evaluation ---

input_dim = X_train.shape[1]

# Baseline model training + profiling
baseline = train_model(build_baseline_cnn_gru(input_dim), X_train, y_train, EPOCHS, BATCH_SIZE)
baseline_stats = evaluate_model(baseline, X_test, y_test) | profile_model(baseline, X_test[:1])
baseline_stats["params"] = baseline.count_params()

# Optimized model training + profiling
optimized = train_model(build_optimized_cnn_gru(input_dim), X_train, y_train, EPOCHS, BATCH_SIZE)
optimized_stats = evaluate_model(optimized, X_test, y_test) | profile_model(optimized, X_test[:1])
optimized_stats["params"] = optimized.count_params()

# Prune, retrain, and quantize
pruned = apply_pruning(build_optimized_cnn_gru(input_dim))
pruned.compile(optimizer=OPTIMIZER_VAL, loss=LOSS_VAL, metrics=['accuracy'])
pruned.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
           callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=0)
quant_model = tfmot.sparsity.keras.strip_pruning(pruned)
quantized_tflite = quantize_model(quant_model)

# Quantized evaluation + profiling
quantized_stats = evaluate_tflite(quantized_tflite, X_test, y_test, quant_model.count_params())
quantized_stats |= profile_tflite_model(quantized_tflite, X_test[0])

# --- Save and Export Results ---

results = pd.DataFrame(
    [baseline_stats, optimized_stats, quantized_stats],
    index=["Baseline CNN+GRU", "Optimized CNN+GRU", "Quantized + Pruned CNN+GRU"]
)

save_results(results.to_dict(orient="index"), "tensorflow_cnn_gru_comparison.csv")
print(results)
