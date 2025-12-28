import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Customer Behavioral Clustering",
    layout="wide"
)

st.title("Customer Behavioral Change Analysis")
st.markdown("""
This application performs **unsupervised customer segmentation**.
It automatically determines the optimal number of clusters using
**Silhouette Analysis**, then applies **K-Means clustering**.
""")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload an Excel file (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file to begin.")
    st.stop()

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df

df = load_data(uploaded_file)

# -------------------- PREVIEW --------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------- FEATURE SELECTION --------------------
st.sidebar.header("Clustering Settings")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns for clustering.")
    st.stop()

features = st.sidebar.multiselect(
    "Select numeric features for clustering",
    options=numeric_cols,
    default=numeric_cols
)

if len(features) < 2:
    st.warning("Please select at least two numeric features.")
    st.stop()

# -------------------- DATA CLEANING --------------------
X = df[features].copy()

# Replace infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

rows_before = len(X)

# Drop rows with missing values
X.dropna(inplace=True)

rows_after = len(X)

# Align original dataframe with cleaned data
df = df.loc[X.index].copy()

if rows_after < 2:
    st.error("Not enough valid rows after cleaning to perform clustering.")
    st.stop()

if rows_before != rows_after:
    st.warning(
        f"{rows_before - rows_after} rows were removed due to missing or invalid values."
    )

# -------------------- SCALING --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- SILHOUETTE ANALYSIS --------------------
st.subheader("Silhouette Analysis")

max_k = min(10, len(X) - 1)
k_range = range(2, max_k + 1)

silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

best_k = k_range[silhouette_scores.index(max(silhouette_scores))]

st.markdown(f"""
### 🟣 Optimal Number of Clusters: **K = {best_k}**
""")

fig, ax = plt.subplots()
ax.plot(k_range, silhouette_scores, marker="o")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score vs K")
st.pyplot(fig)

# -------------------- FINAL KMEANS --------------------
kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans_final.fit_predict(X_scaled)

# -------------------- CLUSTERED DATA --------------------
st.subheader("Clustered Dataset")
st.dataframe(df)

# -------------------- AGGREGATE METRICS --------------------
st.subheader("Cluster-Level Aggregate Metrics")

cluster_summary = (
    df
    .groupby("cluster")[features]
    .mean()
    .round(2)
    .reset_index()
)

st.dataframe(cluster_summary)

# -------------------- VISUAL COMPARISON --------------------
st.subheader("Cluster Comparison")

metric = st.selectbox(
    "Select a metric to visualize",
    features
)

fig2, ax2 = plt.subplots()
ax2.bar(cluster_summary["cluster"], cluster_summary[metric])
ax2.set_xlabel("Cluster")
ax2.set_ylabel(metric.replace("_", " ").title())
ax2.set_title(f"{metric.replace('_', ' ').title()} by Cluster")
st.pyplot(fig2)

# -------------------- DOWNLOAD RESULTS --------------------
st.subheader("Download Results")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Clustered Dataset",
    data=csv,
    file_name="clustered_customers.csv",
    mime="text/csv"
)
