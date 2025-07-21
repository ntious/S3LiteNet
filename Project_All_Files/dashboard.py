import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("Benchmarking Lightweight Neural Models for Edge-Based Anomaly Detection: A Comparative Study Dashboard")

# === Step 1: Load and merge all .csv files in the results folder ===
results_dir = "results"
all_data = []

for filename in os.listdir(results_dir):
    if filename.endswith(".csv"):
        model_name = filename.replace("_comparison.csv", "")
        file_path = os.path.join(results_dir, filename)
        try:
            df = pd.read_csv(file_path)
            df["model"] = model_name
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")  # Normalize early
            all_data.append(df)
        except Exception as e:
            st.warning(f"Failed to read {filename}: {e}")

if not all_data:
    st.error("No results found. Please upload valid .csv files to the 'results/' folder.")
    st.stop()

df = pd.concat(all_data, ignore_index=True)

# === Check Columns ===
st.write("Detected Columns:", df.columns.tolist())

if "dataset" not in df.columns:
    st.error("Missing required column: 'dataset'. Please check your CSV headers.")
    st.stop()

# === Step 2: Filters ===
st.sidebar.header("Filter Options")
models = st.sidebar.multiselect("Select Models", options=df["model"].unique(), default=df["model"].unique())
datasets = st.sidebar.multiselect("Select Datasets", options=df["dataset"].unique(), default=df["dataset"].unique())

filtered_df = df[(df["model"].isin(models)) & (df["dataset"].isin(datasets))]

# === Step 3: Metrics Chart ===
st.subheader("Performance Metrics")
metric = st.selectbox("Choose Metric", ["accuracy", "precision", "recall", "f1", "auc"])

fig = px.bar(filtered_df, x="model", y=metric, color="dataset", barmode="group", height=400)
st.plotly_chart(fig)

# === Step 4: Resource Usage Charts ===
st.subheader("Resource Efficiency")
col1, col2 = st.columns(2)

with col1:
    fig_latency = px.bar(filtered_df, x="model", y="latency", color="dataset", title="Latency (s)")
    st.plotly_chart(fig_latency)

with col2:
    fig_size = px.bar(filtered_df, x="model", y="size_mb", color="dataset", title="Model Size (MB)")
    st.plotly_chart(fig_size)

# === Step 5: Upload new results ===
st.subheader("Upload New .csv Result")
uploaded_file = st.file_uploader("Upload a new model comparison .csv file", type=["csv"])
if uploaded_file is not None:
    try:
        new_model = uploaded_file.name.replace("_comparison.csv", "")
        new_df = pd.read_csv(uploaded_file)
        new_df["model"] = new_model
        new_df.columns = new_df.columns.str.strip().str.lower().str.replace(" ", "_")
        df = pd.concat([df, new_df], ignore_index=True)
        st.success(f"Uploaded and merged results for: {new_model}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
