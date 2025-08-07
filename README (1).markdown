Customer Segmentation for Business
Overview
This project uses K-means clustering to segment mall customers based on annual income and spending score, enabling targeted marketing strategies. It includes data preprocessing, clustering analysis, and an interactive Streamlit dashboard for visualization.
Features

Clustering: Applies K-means to group customers into 5 segments.
Visualization: Scatter plots with silhouette scores to assess cluster quality.
Interactive Dashboard: Streamlit app for exploring cluster statistics and segment insights.

Dataset

Source: Mall Customers by Random on Kaggle.
File: Mall_Customers.csv (200 records with customer ID, gender, age, income, spending score).
Note: The script downloads the dataset via Kaggle API during runtime. A Kaggle API token is required for deployment.

Requirements

Python 3.8+
Libraries: Install via pip install -r requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
kaggle



Setup Instructions

Clone the repository:git clone https://github.com/bhanup23/customer-segmentation
cd customer-segmentation


Install dependencies:pip install -r requirements.txt


Set up Kaggle API:
Download your Kaggle API token from Kaggle (JSON file).
Place the kaggle.json file in ~/.kaggle/ (create the directory if needed) or the project root.
Ensure permissions: chmod 600 ~/.kaggle/kaggle.json.


Run the Streamlit app:streamlit run customer_segmentation.py


Open the URL (e.g., http://localhost:8501) in your browser.

Deployment

Streamlit Community Cloud: Deploy for a live demo:
Push to GitHub.
Connect to Streamlit Community Cloud and select the repository.
Upload kaggle.json to the Streamlit Cloud secrets manager (under "Secrets" in the app settings).
Set customer_segmentation.py as the main script and include requirements.txt.


Live Demo: (https://customer-segmentation-mjw9uy2zkaeuzat3odxhej.streamlit.app).

Project Structure

customer_segmentation.py: Main script for clustering and Streamlit dashboard.
Mall_Customers.csv: Dataset file (downloaded via Kaggle API).
requirements.txt: Dependencies.
clusters.png: Generated visualization.
kaggle.json: Kaggle API token (not included in repo; add manually).

Results

Identifies 5 customer segments (e.g., high spenders, budget shoppers) with a silhouette score ~0.55.
Visualizations highlight income-spending patterns for marketing.

Future Improvements

Add hierarchical clustering or GMM for comparison.
Incorporate additional features (e.g., age) with preprocessing.
Integrate real-time customer data.

Author

Bhanu Pratap




