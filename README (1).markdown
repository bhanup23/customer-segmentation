# Customer Segmentation for Business

## Overview
This project uses K-means clustering to segment mall customers based on annual income and spending score, enabling targeted marketing strategies. It includes data preprocessing, clustering analysis, and an interactive Streamlit dashboard for visualization.

## Features
- **Clustering**: Applies K-means to group customers into 5 segments.
- **Visualization**: Scatter plots with silhouette scores to assess cluster quality.
- **Interactive Dashboard**: Streamlit app for exploring cluster statistics and segment insights.

## Dataset
- **Source**: [Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) by Random on Kaggle.
- **File**: `Mall_Customers.csv` (200 records with customer ID, gender, age, income, spending score).
- **Note**: Download and place the file in the project directory.

## Requirements
- Python 3.8+
- Libraries: Install via `pip install -r requirements.txt`
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - streamlit

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-segmentation.git
   cd customer-segmentation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and place `Mall_Customers.csv` in the project directory.
4. Run the Streamlit app:
   ```bash
   streamlit run customer_segmentation.py
   ```
5. Open the URL (e.g., `http://localhost:8501`) in your browser.

## Deployment
- **Streamlit Community Cloud**: Deploy for a live demo:
  1. Push to GitHub.
  2. Connect to [Streamlit Community Cloud](https://streamlit.io/cloud) and select the repository.
  3. Include `requirements.txt` and dataset.
- **Live Demo**: (Add your URL, e.g., `https://your-customer-segmentation.streamlit.app`).

## Project Structure
- `customer_segmentation.py`: Main script for clustering and Streamlit dashboard.
- `Mall_Customers.csv`: Dataset file (download from Kaggle).
- `requirements.txt`: Dependencies.
- `clusters.png`: Generated visualization.

## Results
- Identifies 5 customer segments (e.g., high spenders, budget shoppers) with a silhouette score ~0.55.
- Visualizations highlight income-spending patterns for targeted marketing.

## Future Improvements
- Add hierarchical clustering for comparison.
- Incorporate more features (e.g., age, gender) with advanced preprocessing.
- Deploy with real-time data integration.

## Author
- Bhanu Pratap
- GitHub: [yourusername](https://github.com/yourusername)
- Email: bhanu2208@iitk.ac.in

## License
MIT License