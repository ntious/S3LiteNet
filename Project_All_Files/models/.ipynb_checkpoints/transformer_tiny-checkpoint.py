
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pandas as pd\nfrom logger.result_logger import save_results
import numpy as np
from config import EPOCHS, BATCH_SIZE
from data.preprocess import load_and_preprocess
from train_eval.train import train_model
from train_eval.metrics import evaluate_model
from benchmark.resource_profiler import profile_model

X_train, X_test, y_train, y_test = load_and_preprocess()

def transformer_block(embed_dim, num_heads, ff_dim):
    inputs = tf.keras.Input(shape=(1, embed_dim))
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(0.1)(attention_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(embed_dim),
    ])
    ffn_output = ffn(out1)
    ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
    return tf.keras.Model(inputs, tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output))

def build_baseline_transformer(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((1, input_dim))(inputs)
    x = transformer_block(input_dim, num_heads=2, ff_dim=64)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)

def build_optimized_transformer(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((1, input_dim))(inputs)
    x = transformer_block(input_dim, num_heads=1, ff_dim=32)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)

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

input_dim = X_train.shape[1]
baseline = train_model(build_baseline_transformer(input_dim), X_train, y_train, EPOCHS, BATCH_SIZE)
optimized = train_model(build_optimized_transformer(input_dim), X_train, y_train, EPOCHS, BATCH_SIZE)

pruned = apply_pruning(build_optimized_transformer(input_dim))
pruned.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
pruned.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=0)
quant_model = tfmot.sparsity.keras.strip_pruning(pruned)
quantized_tflite = quantize_model(quant_model)

baseline_stats = evaluate_model(baseline, X_test, y_test) | profile_model(baseline, X_test[:1])
optimized_stats = evaluate_model(optimized, X_test, y_test) | profile_model(optimized, X_test[:1])
quantized_stats = evaluate_tflite(quantized_tflite, X_test, y_test)

results = pd.DataFrame([baseline_stats, optimized_stats, quantized_stats],
                       index=["Baseline Transformer", "Optimized Transformer", "Quantized + Pruned Transformer"])
save_results(results.to_dict(orient="index"), "tensorflow_transformer_comparison.csv")
print(results)
