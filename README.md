# EEL: Embedding Exploration Lab

## Overview

The Embedding Exploration Lab (EEL) is a tool designed for visualizing high-dimensional embeddings in a 3D space. This class provides functionality to load embeddings from a FAISS index, reduce their dimensionality using PCA and/or t-SNE, and visualize them in an interactive 3D plot.

## Installation

To use EEL, you need to have the following Python packages installed:
- `faiss`
- `numpy`
- `plotly`
- `sklearn`
- `tqdm`
- `umap-learn`
- `datasets`

You can install these packages via pip:

```bash
pip install faiss-cpu numpy plotly scikit-learn tqdm umap-learn datasets
```

## Usage

First, initialize the EEL class with the paths to your FAISS index and dataset:

```python
from eel import EEL

visualizer = EEL(index_path="path/to/index", dataset_path="path/to/dataset")
```
### Class: EEL

#### Parameters

`index_path` (str): Path to the FAISS index file.  
`dataset_path` (str): Path to the dataset containing labels.

#### Attributes

`index_path` (str): Path to the FAISS index file.  
`dataset_path` (str): Path to the dataset containing labels.  
`index` (faiss.Index or None): Loaded FAISS index.  
`dataset` (datasets.Dataset or None): Loaded dataset containing labels.  
`vectors` (np.ndarray or None): Extracted vectors from the FAISS index.  
`reduced_vectors` (np.ndarray or None): Dimensionality-reduced vectors.  
`labels` (list of str or None): Labels from the dataset.

#### Methods

`load_index()`
Load the FAISS index from the specified file path.

Returns:
- `self (EEL)`: The instance itself, allowing for method chaining.

`load_dataset(column: str = "document")`
Load the Dataset containing labels from the specified file path.

Parameters:
- `column` (str, optional): The column of the split corresponding to the embeddings stored in the index. Default is 'document'.

Returns:
- `self (EEL)`: The instance itself, allowing for method chaining.

`extract_vectors()`
Extract all vectors from the loaded FAISS index.

Returns:
- `self (EEL)`: The instance itself, allowing for method chaining.

Raises:
- `ValueError`: If the index has not been loaded yet.
- `RuntimeError`: If there's an issue with vector extraction.

`reduce_dimensionality(method: str = "umap", pca_components: int = 50, final_components: int = 3, random_state: int = 42)`
Reduce dimensionality of the extracted vectors with dynamic progress tracking.

Parameters:
- `method` (str, optional): The method to use for dimensionality reduction. Options: 'pca', 'umap', 'pca_umap'. Default is 'umap'.
- `pca_components` (int, optional): Number of components for PCA (used in 'pca' and 'pca_umap'). Default is 50.
- `final_components` (int, optional): Final number of components (3 for 3D visualization). Default is 3.
- `random_state` (int, optional): Random state for reproducibility. Default is 42.

Returns:
- `self (EEL)`: The instance itself, allowing for method chaining.

Raises:
- `ValueError`: If vectors have not been extracted yet or if an invalid method is specified.

`create_plot(title: str = "3D Visualization of Embeddings", point_size: int = 3)`
Generate a 3D scatter plot of the reduced vectors with labels.

Parameters:
- `title` (str, optional): The title of the plot. Default is '3D Visualization of Embeddings'.
- `point_size` (int, optional): The size of the markers in the scatter plot. Default is 3.

Returns:
- `go.Figure`: The generated 3D scatter plot.

Raises:
- `ValueError`: If vectors have not been reduced yet.

`visualize(method: str = "tsne", pca_components: int = 50, final_components: int = 3, random_state: int = 42, title: str = "3D Visualization of Embeddings", point_size: int = 3, save_html: bool = False, html_file_name: str = "embedding_visualization.html")`
Full pipeline: load index, extract vectors, reduce dimensionality, and visualize.

Parameters:
- `method` (str, optional): The dimensionality reduction method to use. Default is 'tsne'.
- `pca_components` (int, optional): The number of components to keep when using PCA for dimensionality reduction. Default is 50.
- `final_components` (int, optional): The number of final components to visualize. Default is 3.
- `random_state` (int, optional): The random state for reproducibility. Default is 42.
- `title` (str, optional): The title of the visualization plot. Default is '3D Visualization of Embeddings'.
- `point_size` (int, optional): The size of the points in the visualization plot. Default is 3.
- `save_html` (bool, optional): Whether to save the visualization as an HTML file. Default is False.
- `html_file_name` (str, optional): The name of the HTML file to save. Default is 'embedding_visualization.html'.