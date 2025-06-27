import os
import streamlit as st

# Step 1: Create the ~/.kaggle folder
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# Step 2: Write the kaggle.json file from Streamlit secrets
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    f.write('{"username":"%s","key":"%s"}' % (
        st.secrets["kaggle"]["username"],
        st.secrets["kaggle"]["key"]
    ))

# Step 3: Set correct permissions
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Step 4: Now safely import kaggle (after credentials exist)
import kaggle



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st
import os


# Load or download dataset
@st.cache_data
def load_data():
    dataset_path = 'Mall_Customers.csv'
    if not os.path.exists(dataset_path):
        st.warning("Dataset not found locally. Attempting to download from Kaggle...")
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('vjchoudhary7/customer-segmentation-tutorial-in-python', path='.', unzip=True)
            st.success("Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download dataset from Kaggle: {e}. Please upload Mall_Customers.csv manually.")
            return None
    df = pd.read_csv(dataset_path)
    df = df.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_Score'})
    return df

# Preprocess data
def preprocess_data(df):
    if df is None:
        st.error("No data available for preprocessing.")
        return None, None
    X = df[['Annual_Income', 'Spending_Score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Apply K-means clustering
def apply_clustering(X_scaled, n_clusters=5):
    if X_scaled is None:
        return None, None, None
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, clusters)
    return clusters, kmeans, silhouette_avg

# Visualize clusters
def plot_clusters(X, clusters, silhouette_avg):
    if X is None or clusters is None:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clusters, palette='deep', s=100)
    plt.title(f'Customer Segments (Silhouette Score: {silhouette_avg:.2f})')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.legend(title='Cluster')
    plt.savefig('clusters.png')
    plt.close()

# Streamlit Dashboard
def main():
    st.title('Customer Segmentation Dashboard')
    st.write('Segment customers using K-means clustering and explore insights.')

    # Load and preprocess data
    df = load_data()
    if df is None:
        st.stop()
    X_scaled, scaler = preprocess_data(df)
    
    # Apply clustering
    clusters, kmeans, silhouette_avg = apply_clustering(X_scaled)
    if clusters is None:
        st.stop()
    plot_clusters(X_scaled, clusters, silhouette_avg)
    
    # Sidebar for cluster selection
    st.sidebar.header('Cluster Insights')
    selected_cluster = st.sidebar.selectbox('Select Cluster', range(5))
    
    # Display cluster analysis
    st.header('Cluster Visualization')
    st.image('clusters.png')
    st.write(f'Silhouette Score: {silhouette_avg:.2f} (indicates cluster quality)')
    
    cluster_data = pd.DataFrame({'Annual_Income': X_scaled[:, 0], 'Spending_Score': X_scaled[:, 1], 'Cluster': clusters})
    selected_data = cluster_data[cluster_data['Cluster'] == selected_cluster]
    st.write(f'Cluster {selected_cluster} Statistics:')
    st.write(selected_data.describe())
    
    # Add segment description
    segments = {
        0: 'Low Income, Low Spenders',
        1: 'High Income, High Spenders',
        2: 'Moderate Income, Low Spenders',
        3: 'Low Income, High Spenders',
        4: 'Moderate Income, Moderate Spenders'
    }
    st.write(f'Segment Description: {segments[selected_cluster]}')

if __name__ == '__main__':
    main()
