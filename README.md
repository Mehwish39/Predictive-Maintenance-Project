# Predictive Maintenance using Machine Learning
This project applies machine learning algorithms for predictive maintenance in industrial environments, focusing on detecting potential equipment failures before they happen. 
The datasets used in this project include the Industrial Internet of Things (IIoT) dataset and a manual inspection dataset. 
The goal is to apply unsupervised learning techniques such as K-Means, Isolation Forest, and Autoencoders to detect anomalies in the data, which may indicate impending equipment failures. 

## Datasets
- **IIoT Data**: Real-time sensor data collected from industrial machines.
- **Manual Inspection Data**: A larger dataset that is prone to human error, consisting of manually recorded machine inspection data.

## Algorithms Used
1. **K-Means Clustering**: Classifies data into clusters and detects anomalies based on distance from cluster centroids.
2. **Isolation Forest**: Identifies outliers by recursively partitioning the data.
3. **Autoencoders**: Detects anomalies based on high reconstruction errors in the data.
4. **Gaussian Mixture Model**: Models data as a mixture of multiple Gaussian distributions and detects anomalies.
5. **DBSCAN**: A density-based clustering algorithm that identifies clusters and outliers.

## Project Files
Data Sets  
- iiot_30min_norm.csv  # IIoT data  
- manual_30min_norm.csv  # Manual inspection data  
Jupyter Notebooks  
- Predictive_Maintenance_Models.ipynb  # Jupyter notebook with the machine learning models  


## Requirements
- Python 3.7+
- `pandas` for data manipulation
- `numpy` for numerical operations
- `scikit-learn` for machine learning algorithms
- `tensorflow` for building autoencoders
- `matplotlib` for visualizations

## Install the necessary libraries by running:  
pip install -r requirements.txt  

## How to Use
Clone the repository:
  git clone https://github.com/yourusername/PredictiveMaintenance.git  
Load the data in the Jupyter notebook:
  iiot_data = pd.read_csv('data/iiot_30min_norm.csv')  
  manual_data = pd.read_csv('data/manual_30min_norm.csv')  
Open and run the Predictive_Maintenance_Models.ipynb notebook to train the models on the datasets.

## Results
The models are evaluated using metrics such as Precision, Recall, F1-Score, and Cohen's Kappa.
Isolation Forest and DBSCAN showed the best results on the IIoT dataset, while K-Means and Autoencoders provided balanced results across both datasets.

