import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    df = df.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_Score'})
    return df

# Preprocess data
def preprocess_data(df):
    # Select relevant features
    X = df[['Annual_Income', 'Spending_Score']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# Apply K-means clustering
def apply_clustering(X_scaled, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, clusters)
    return clusters, kmeans, silhouette_avg

# Visualize clusters
def plot_clusters(X, clusters, silhouette_avg):
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
    X_scaled, scaler = preprocess_data(df)
    
    # Apply clustering
    clusters, kmeans, silhouette_avg = apply_clustering(X_scaled)
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