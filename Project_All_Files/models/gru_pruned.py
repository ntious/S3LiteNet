# GRU Benchmarking Script
# Compares Baseline, Optimized, and Quantized + Pruned GRU models
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pandas as pd
import numpy as np
from config import EPOCHS, BATCH_SIZE, OPTIMIZER_VAL, LOSS_VAL 
from data.preprocess import load_and_preprocess
from train_eval.train import train_model
from train_eval.metrics import evaluate_model
from benchmark.resource_profiler import profile_model, profile_tflite_model
from logger.result_logger import save_results

# Load preprocessed data
X_train, X_test, y_train, y_test = load_and_preprocess()

# --- GRU Model Definitions ---

# Baseline GRU: single GRU layer followed by a dense output
def build_baseline_gru(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((1, input_shape), input_shape=(input_shape,)),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Optimized GRU: smaller GRU layer with global pooling
def build_optimized_gru(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((1, input_shape), input_shape=(input_shape,)),
        tf.keras.layers.GRU(32, return_sequences=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Pruning utility: progressively sparsifies weights during training
def apply_pruning(model):
    schedule = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                     final_sparsity=0.5,
                                                     begin_step=0,
                                                     end_step=1000)
    return tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=schedule)

# Quantization utility: compresses the model using post-training quantization
def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Allow fallback to TensorFlow ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,         # default TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS            # enable TF fallback ops
    ]

    # Disable lowering of tensor list ops
    converter._experimental_lower_tensor_list_ops = False

    # Optional: Post-training dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    return converter.convert()

# Evaluation wrapper for TFLite models
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

# Extract feature dimension
input_dim = X_train.shape[1]

# Train and evaluate baseline GRU
baseline = train_model(build_baseline_gru(input_dim), X_train, y_train, EPOCHS, BATCH_SIZE)
baseline_stats = evaluate_model(baseline, X_test, y_test)
baseline_stats |= profile_model(baseline, X_test[:1])
baseline_stats["params"] = baseline.count_params()

# Train and evaluate optimized GRU
optimized = train_model(build_optimized_gru(input_dim), X_train, y_train, EPOCHS, BATCH_SIZE)
optimized_stats = evaluate_model(optimized, X_test, y_test)
optimized_stats |= profile_model(optimized, X_test[:1])
optimized_stats["params"] = optimized.count_params()

# Prune and retrain the optimized GRU
pruned = apply_pruning(build_optimized_gru(input_dim))
pruned.compile(optimizer=OPTIMIZER_VAL, loss=LOSS_VAL, metrics=['accuracy'])
pruned.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
           callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=0)

# Strip pruning wrappers and quantize the model
quant_model = tfmot.sparsity.keras.strip_pruning(pruned)
quantized_tflite = quantize_model(quant_model)

# Evaluate and profile the quantized model
quantized_stats = evaluate_tflite(quantized_tflite, X_test, y_test, quant_model.count_params())
quantized_stats |= profile_tflite_model(quantized_tflite, X_test[0])

# --- Save Results ---
results = pd.DataFrame(
    [baseline_stats, optimized_stats, quantized_stats],
    index=["Baseline GRU", "Optimized GRU", "Quantized + Pruned GRU"])

save_results(results.to_dict(orient="index"), "tensorflow_gru_comparison.csv")
print(results)
