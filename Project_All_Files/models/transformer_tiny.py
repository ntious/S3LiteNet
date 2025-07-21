# Transformer Model
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd
from config import RANDOM_STATE, EPOCHS, OPTIMIZER_VAL, LOSS_VAL,BATCH_SIZE  
from data.preprocess import load_and_preprocess
from train_eval.metrics import evaluate_model
from benchmark.resource_profiler import profile_model, profile_tflite_model
from logger.result_logger import save_results

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess()

# Define transformer architecture
def build_baseline_transformer(input_dim, embed_dim=64):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    x = tf.keras.layers.Reshape((1, embed_dim))(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embed_dim)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

def build_optimized_transformer(input_dim):
    return build_baseline_transformer(input_dim, embed_dim=32)

# Quantization utility
def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()

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

# Run evaluation
input_dim = X_train.shape[1]

baseline = build_baseline_transformer(input_dim)
optimized = build_optimized_transformer(input_dim)

def train_and_eval(model):
    model.compile(optimizer=OPTIMIZER_VAL, loss=LOSS_VAL, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    stats = evaluate_model(model, X_test, y_test)
    profile = profile_model(model, X_test[:1])
    return stats | profile | {"params": model.count_params()}

baseline_stats = train_and_eval(baseline)
optimized_stats = train_and_eval(optimized)

quant_model = optimized
quant_model.compile(optimizer=OPTIMIZER_VAL, loss=LOSS_VAL, metrics=['accuracy'])
quant_model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)
quantized_tflite = quantize_model(quant_model)
quantized_stats = evaluate_tflite(quantized_tflite, X_test, y_test, quant_model.count_params())
quantized_stats |= profile_tflite_model(quantized_tflite, X_test[0])

# Save results
results = pd.DataFrame(
    [baseline_stats, optimized_stats, quantized_stats],
    index=["Baseline Transformer", "Optimized Transformer", "Quantized Transformer"]
)
save_results(results.to_dict(orient="index"), "tensorflow_transformer_comparison.csv")
print(results)
