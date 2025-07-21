import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

# Load and preprocess
df = pd.read_csv("data/Combined_CIC_IoT_DIAD_2024.csv")
df.drop(columns=["Src IP", "Dst IP", "Flow ID", "Timestamp", "Attack Name"], inplace=True, errors='ignore')
X = df.drop(columns=["Label"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
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

def train_and_eval(model, X_train, y_train, X_test, y_test, epochs=5):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "params": model.count_params()
    }

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
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, preds),
        "params": "quantized"
    }

# Run all
input_dim = X_train.shape[1]
baseline = build_baseline_cnn(input_dim)
optimized = build_optimized_cnn(input_dim)
pruned = apply_pruning(build_optimized_cnn(input_dim))
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

baseline_stats = train_and_eval(baseline, X_train, y_train, X_test, y_test)
optimized_stats = train_and_eval(optimized, X_train, y_train, X_test, y_test)
pruned.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
pruned.fit(X_train, y_train, epochs=5, batch_size=64, callbacks=callbacks, verbose=0)
quant_model = tfmot.sparsity.keras.strip_pruning(pruned)
quantized_tflite = quantize_model(quant_model)
quantized_stats = evaluate_tflite(quantized_tflite, X_test, y_test)

# Save results
results = pd.DataFrame([baseline_stats, optimized_stats, quantized_stats],
                       index=["Baseline CNN", "Optimized CNN", "Quantized + Pruned CNN"])
results.to_csv("tensorflow_cnn_comparison.csv")
print(results)
