import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Behavior Analysis", layout="wide")

# Title
st.title("📊 Customer Behavior Analysis")
st.write("Segment customers into meaningful groups using machine learning clustering.")

# Load dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample dataset...")
    # sample synthetic dataset
    np.random.seed(42)
    df = pd.DataFrame({
        "Age": np.random.randint(18, 60, 300),
        "Income": np.random.randint(20000, 120000, 300),
        "SpendingScore": np.random.randint(1, 100, 300),
        "VisitsPerMonth": np.random.randint(1, 20, 300)
    })

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select features
features = st.multiselect(
    "Select features for clustering",
    df.columns,
    default=df.columns
)

if len(features) < 2:
    st.warning("Select at least 2 features for clustering.")
    st.stop()

X = df[features]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose clusters
k = st.slider("Select number of clusters (K)", 2, 10, 4)

# KMeans
model = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

st.subheader("Clustered Data")
st.dataframe(df.head())

# PCA visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(components[:, 0], components[:, 1], c=clusters, cmap="viridis")
ax.set_title("Customer Segments (PCA Projection)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

st.pyplot(fig)

# Cluster insights
st.subheader("Cluster Insights")

for i in range(k):
    st.markdown(f"### Cluster {i}")
    st.write(df[df["Cluster"] == i].describe())
