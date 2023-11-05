import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import plotly.express as px
from scipy.spatial import distance
from collections import Counter
import itertools

class DataHandler:
    """
    Handles data operations including reading and saving data.
    """
    def __init__(self):
        self.data = None

    def read_excel(self, file_path, usecols=None):
        """
        Reads data from an Excel file into a pandas DataFrame.
        """
        try:
            self.data = pd.read_excel(file_path, usecols=usecols)
        except Exception as e:
            print(f"An error occurred while reading the Excel file: {e}")

    def save_parquet(self, file_name):
        """
        Saves the pandas DataFrame to a Parquet file.
        """
        try:
            self.data.to_parquet(file_name)
        except Exception as e:
            print(f"An error occurred while saving to Parquet file: {e}")

class TextEmbedder:
    """
    Handles the embedding of text data.
    """
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text_series):
        """
        Generates embeddings for the given text series using the Sentence Transformer model.
        """
        return self.model.encode(text_series.tolist())

class Clusterer:
    """
    Handles clustering operations.
    """
    def __init__(self, n_clusters=20):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.centroids = None

    def fit(self, embeddings):
        """
        Fits the KMeans clustering model to the embeddings and finds cluster centroids.
        """
        self.centroids = self.kmeans.fit_predict(embeddings)
        return self.centroids

class DimensionReducer:
    """
    Handles dimensionality reduction.
    """
    def __init__(self, n_components=2):
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)

    def reduce_dimensions(self, embeddings, labels):
        """
        Reduces the dimensionality of the embeddings to the number of specified components.
        """
        return self.lda.fit_transform(embeddings, labels)

class Plotter:
    """
    Handles the creation of plots.
    """
    def scatter_plot(self, df, x_col, y_col, color_col, title="Scatter Plot"):
        """
        Creates a scatter plot with the given DataFrame and columns.
        """
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        fig.show()

    def treemap(self, df, path_col, size_col, title="Treemap"):
        """
        Creates a treemap with the given DataFrame and columns.
        """
        fig = px.treemap(df, path=[path_col], values=size_col, title=title)
        fig.show()


class ClusterNamer:
    """
    Generates names for clusters based on the most common words in the text data.
    """
    def __init__(self):
        pass

    def name_clusters(self, data, cluster_column, text_column):
        """
        Generates and assigns names to the clusters based on the task names.
        """
        cluster_names = {}
        for cluster in sorted(data[cluster_column].unique()):
            task_names = data[data[cluster_column] == cluster][text_column]
            words = list(itertools.chain(*[name.split() for name in task_names]))
            most_common_words = [word for word, word_count in Counter(words).most_common(3)]
            cluster_names[cluster] = ' '.join(most_common_words)
        return cluster_names

class DataFrameMerger:
    """
    Handles the merging of dataframes.
    """
    @staticmethod
    def merge_with_counts(data, cluster_column):
        """
        Merges the DataFrame with a count of occurrences per cluster.
        """
        count_series = data[cluster_column].value_counts().rename('Count')
        return data.merge(count_series, left_on=cluster_column, right_index=True)

# Now, let's instantiate and use these classes.
data_handler = DataHandler()
data_handler.read_excel('/home/oliver/ChatDev/tools_data.xlsx', usecols=[0, 1, 2])
data_handler.save_parquet('/mnt/data/Data_pd.parquet')

text_embedder = TextEmbedder('BAAI/bge-small-en-v1.5')
embeddings = text_embedder.embed_text(data_handler.data['Task Name'])

clusterer = Clusterer(n_clusters=20)
cluster_labels = clusterer.fit(embeddings)
data_handler.data['Cluster'] = cluster_labels

dimension_reducer = DimensionReducer(n_components=2)
reduced_embeddings = dimension_reducer.reduce_dimensions(embeddings, cluster_labels)
data_handler.data['Reduced Embedding'] = list(reduced_embeddings)
data_handler.save_parquet('/mnt/data/Data_pd.parquet')

plotter = Plotter()
scatter_data = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
scatter_data['Cluster'] = cluster_labels
plotter.scatter_plot(scatter_data, 'x', 'y', 'Cluster', 'Scatter plot of Reduced Embeddings')

cluster_namer = ClusterNamer()
cluster_names = cluster_namer.name_clusters(data_handler.data, 'Cluster', 'Task Name')
for cluster, name in cluster_names.items():
    data_handler.data.loc[data_handler.data['Cluster'] == cluster, 'Cluster Name'] = name
data_handler.save_parquet('/mnt/data/Data_pd.parquet')

# Finally, merge the DataFrame with the counts and display the table.
data_with_counts = DataFrameMerger.merge_with_counts(data_handler.data, 'Cluster')
data_with_counts.to_parquet('/mnt/data/Data_pd.parquet')

# Displaying the table with cluster names and counts.
print(data_with_counts[['Cluster', 'Cluster Name', 'Count']])

# Usage of the Plotter class for a treemap
plotter = Plotter()

# Assuming 'data_with_counts' is a DataFrame with a 'Cluster Name' and 'Count' columns
# which was created in the previous steps.
plotter.treemap(data_with_counts, path_col='Cluster Name', size_col='Count', title="Treemap of Clusters")
