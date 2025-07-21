
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pandas as pd
import numpy as np
from config import EPOCHS, BATCH_SIZE
from data.preprocess import load_and_preprocess
from train_eval.train import train_model
from train_eval.metrics import evaluate_model
from benchmark.resource_profiler import profile_model

X_train, X_test, y_train, y_test = load_and_preprocess()

def build_baseline_cnn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_optimized_cnn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def apply_pruning(model):
    schedule = tfmot.sparsity.keras.PolynomialDecay(0.0, 0.5, 0, 1000)
    return tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=schedule)

def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()

def evaluate_tflite(tflite_model, X_test, y_test):
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
    return evaluate_model(lambda x: np.array([[int(preds[i])]])[0], X_test, y_test) | {"params": "quantized"}

# Run benchmarks
input_dim = X_train.shape[1]
baseline = train_model(build_baseline_cnn(input_dim), X_train, y_train, EPOCHS, BATCH_SIZE)
optimized = train_model(build_optimized_cnn(input_dim), X_train, y_train, EPOCHS, BATCH_SIZE)

pruned = apply_pruning(build_optimized_cnn(input_dim))
pruned.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
pruned.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=0)
quant_model = tfmot.sparsity.keras.strip_pruning(pruned)
quantized_tflite = quantize_model(quant_model)

# Evaluate all
baseline_stats = evaluate_model(baseline, X_test, y_test) | profile_model(baseline, X_test[:1])
optimized_stats = evaluate_model(optimized, X_test, y_test) | profile_model(optimized, X_test[:1])
quantized_stats = evaluate_tflite(quantized_tflite, X_test, y_test)

# Save results
results = pd.DataFrame([baseline_stats, optimized_stats, quantized_stats],
                       index=["Baseline CNN", "Optimized CNN", "Quantized + Pruned CNN"])
results.to_csv("results/tensorflow_cnn_comparison.csv")
print(results)
