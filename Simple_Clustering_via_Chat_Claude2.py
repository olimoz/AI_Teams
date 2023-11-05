"""
Clustering pipeline modules and workflow
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
import plotly.express as px
from collections import Counter
import numpy as np
import logging

log = logging.getLogger(__name__)

class DataLoader:
    """Read and extract data"""
    
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)
        
    @property
    def task_data(self):
        return self.data[['Task Name', 'Name', 'Link']]

class EmbeddingModel:
    """Generate embeddings from text"""
    
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, data):
        return data['Task Name'].apply(self.model.encode)
    
class ClusteringModel:
    """Cluster embeddings with K-Means"""
    
    def __init__(self, n_clusters=20):
        self.model = KMeans(n_clusters=n_clusters)
        
    def fit(self, data):
        self.model.fit(data)
        
    def predict(self, data):
        return self.model.predict(data)
    
    def score(self, data):
        return silhouette_score(data, self.predict(data))
        
class DimensionReducer:
    """Reduce dimensionality with LDA"""
    
    def __init__(self, n_components=2):
        self.model = LinearDiscriminantAnalysis(n_components=n_components)
        
    def fit(self, data):
        self.model.fit(data)
        
    def transform(self, data):
        return self.model.transform(data)
    
class Dataframe:
    """Main dataframe object"""
    
    def __init__(self, df):
        self.df = df
        
    def update(self, key, value):
        self.df[key] = value

class ClusterAnalyzer:
    """Analyze clusters and extract insights"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        
    def get_centroids(self):
        return self.df.df.groupby('cluster')['embedding'].agg('mean').reset_index()
    
    def find_closest(self, centroids, n=10):
        closest = pd.DataFrame()
        for _, row in centroids.iterrows():
            cluster_rows = self.df.df[self.df.df['cluster']==row['cluster']] 
            dists = cluster_rows['embedding'].apply(lambda x: np.linalg.norm(x-row['embedding']))
            closest = closest.append(cluster_rows.nlargest(n, 'dists'))
        return closest 
    
    def get_cluster_names(self, closest):
        names = []
        for cluster in closest['cluster'].unique():
            tasks = closest[closest['cluster']==cluster]['Task Name']
            freq = Counter([word for name in tasks for word in name.split()])
            names.append(max(freq, key=freq.get))
        return names
        
class Plotter:
    """Visualize clusters and data"""
    
    @staticmethod
    def plot_clusters(dataframe):
        fig = px.scatter(dataframe.df, x='x', y='y', color='cluster')
        fig.update_traces(textposition='top center')
        fig.show()
        
    @staticmethod
    def plot_centroids(centroids, labels):
        data = {'x': [v[0] for v in centroids['embedding']],
                'y': [v[1] for v in centroids['embedding']],
                'labels': labels}
        fig = px.scatter(data, x='x', y='y', text='labels')
        fig.show()

class ClusteringWorkflow:
    """Orchestrate clustering pipeline"""
    
    def __init__(self, datafile):
        self.loader    = DataLoader(datafile)
        self.model     = EmbeddingModel('BAAI/bge-small-en-v1.5')
        self.kmeans    = ClusteringModel()
        self.reducer   = DimensionReducer()
        self.dataframe = Dataframe(self.loader.task_data)
        self.analyzer  = ClusterAnalyzer(self.dataframe)
        self.plotter   = Plotter()
        
    def run(self):
        
        # Generate embeddings
        log.info('Generating embeddings')
        embeddings = self.model.encode(self.dataframe.df)
        self.dataframe.update('embedding', embeddings)
        
        # Cluster embeddings
        log.info('Clustering embeddings')
        self.kmeans.fit(self.dataframe.df['embedding'])
        self.dataframe.update('cluster', self.kmeans.predict(self.dataframe.df['embedding']))
        
        # Reduce dimensions
        log.info('Reducing dimensions')
        reduced = self.reducer.fit_transform(self.dataframe.df['embedding'])
        self.dataframe.update('x', [v[0] for v in reduced])
        self.dataframe.update('y', [v[1] for v in reduced])
        
        # Visualize clusters
        log.info('Plotting clusters')
        self.plotter.plot_clusters(self.dataframe)
        
        # Get centroids
        log.info('Getting centroids')
        centroids = self.analyzer.get_centroids()
        
        # Find closest to centroids
        log.info('Finding closest to centroids')
        closest = self.analyzer.find_closest(centroids)
        
        # Get cluster names
        log.info('Generating cluster names')
        names = self.analyzer.get_cluster_names(closest)
        
        # Plot centroids
        log.info('Plotting centroid labels')
        self.plotter.plot_centroids(centroids, names)
        
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    workflow = ClusteringWorkflow('data.xlsx')
    workflow.run()