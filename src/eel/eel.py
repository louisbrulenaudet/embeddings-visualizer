# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Type,
    Tuple,
    Union,
    Mapping,
    TypeVar,
    Callable,
    Optional,
    Sequence,
)

import datasets
import faiss
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots

from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from umap import UMAP

from eel._logger import (
    Logger,
    TqdmToLogger
)
from eel._decorators import (
    memory,
    timer
)

logger = Logger()


class EEL:
    """
    A class for Embedding Exploration Lab, visualizing high-dimensional embeddings in 3D space.

    This class provides functionality to load embeddings from a FAISS index,
    reduce their dimensionality using PCA and/or t-SNE, and visualize them
    in an interactive 3D plot.

    Parameters
    ----------
    index_path : str
        Path to the FAISS index file.

    dataset_path : str
        Path to the dataset containing labels.

    Attributes
    ----------
    index_path : str
        Path to the FAISS index file.
    
    dataset_path : str
        Path to the dataset containing labels.
    
    index : faiss.Index or None
        Loaded FAISS index.
    
    dataset : datasets.Dataset or None
        Loaded dataset containing labels.
    
    vectors : np.ndarray or None
        Extracted vectors from the FAISS index.
    
    reduced_vectors : np.ndarray or None
        Dimensionality-reduced vectors.
    
    labels : list of str or None
        Labels from the dataset.

    Methods
    -------
    load_index() -> 'EmbeddingVisualizer':
        Load the FAISS index from the specified file path.
    
    load_dataset() -> 'EmbeddingVisualizer':
        Load the dataset containing labels from the specified file path.
    
    extract_vectors() -> 'EmbeddingVisualizer':
        Extract all vectors from the loaded FAISS index.
    
    reduce_dimensionality(
        method: str = "umap",
        pca_components: int = 50,
        final_components: int = 3,
        random_state: int = 42
    ) -> 'EmbeddingVisualizer':
        Reduce dimensionality of the extracted vectors with dynamic progress tracking.
    
    plot_3d() -> None:
        Generate a 3D scatter plot of the reduced vectors with labels.

    Examples
    --------
    >>> visualizer = EmbeddingVisualizer(index_path="path/to/index", dataset_path="path/to/dataset")
    >>> visualizer.load_index().load_dataset().extract_vectors()
    >>> visualizer.reduce_dimensionality(method="pca_umap", pca_components=50, final_components=3)
    >>> visualizer.plot_3d()
    """
    def __init__(
        self, 
        index_path: str,
        dataset_path: str
    ):
        self.index_path: str = index_path
        self.dataset_path: str = dataset_path
        self.index = None
        self.dataset = None
        self.vectors = None
        self.reduced_vectors = None


    @memory(print_report=True)
    @timer(print_time=True)
    def load_index(
        self
    ):
        """
        Load the FAISS index from the specified file path.

        Returns
        -------
        self : EmbeddingVisualizer
            The instance itself, allowing for method chaining.
        """
        self.index = faiss.read_index_binary(
            self.index_path
        )

        return self

    
    def load_dataset(
        self,
        column: str = "document"
    ):
        """
        Load the Dataset containing labels from the specified file path.

        Parameters
        ----------
        column : str, optional
            The column of the split corresponding to the embeddings stored in 
            the index. Default is 'document'.

        Returns
        -------
        self : datasets.Dataset
            The instance itself, allowing for method chaining.
        """
        self.dataset = datasets.load_from_disk(
            dataset_path=self.dataset_path
        )

        self.labels = self.dataset[column]

        return self


    @memory(print_report=True)
    @timer(print_time=True)
    def extract_vectors(
        self
    ):
        """
        Extract all vectors from the loaded FAISS index.

        This method should be called after `load_index()`.

        Returns
        -------
        self : EmbeddingVisualizer
            The instance itself, allowing for method chaining.

        Raises
        ------
        ValueError
            If the index has not been loaded yet.

        RuntimeError
            If there's an issue with vector extraction.
        """
        global logger

        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")

        if isinstance(self.index, faiss.IndexBinaryFlat):
            # Handle binary index
            num_vectors = self.index.ntotal
            dimension = self.index.d
            code_size = dimension // 8  # Binary vectors are typically in bits, hence // 8 for bytes
            
            logger.info(f"Expected dimension (bits): {dimension}")
            logger.info(f"Index total vectors: {num_vectors}")
            logger.info(f"Index code size (bytes): {code_size}")

            # Binary vectors are stored in uint8 arrays
            self.vectors = np.empty(
                (num_vectors, code_size), 
                dtype=np.uint8
            )

            logger.info(f"Initialized binary vectors array with shape: {self.vectors.shape}")
            
            try:
                for i in tqdm(range(num_vectors)):
                    self.vectors[i] = self.index.reconstruct(i)
            
            except AssertionError:
                raise RuntimeError(
                    f"Dimension mismatch error. Expected dimension (bits): {dimension}, "
                    f"Index total vectors: {num_vectors}, "
                    f"Index code size (bytes): {code_size}. "
                    "The index might be corrupted or incompatible."
                )

            except AttributeError:
                raise RuntimeError(
                    "The loaded index doesn't support binary vector reconstruction."
                    "Make sure you're using an appropriate index type."
                )

            except Exception as e:
                raise RuntimeError(
                    f"Error extracting vectors: {str(e)}"
                )

        else:
            num_vectors = self.index.ntotal
            dimension = self.index.d
            code_size = getattr(self.index, "code_size", None)

            logger.info(f"Expected dimension: {dimension}")
            logger.info(f"Index total vectors: {num_vectors}")
            logger.info(f"Index code size: {code_size}")

            if code_size is not None and code_size != dimension:
                raise RuntimeError(
                    f"Dimension mismatch error. Expected dimension: {dimension}, "
                    f"Index total vectors: {num_vectors}, "
                    f"Index code size: {code_size}. "
                    "The index might be corrupted or incompatible."
                )

            self.vectors = np.empty(
                (num_vectors, dimension), 
                dtype=np.float32
            )

            logger.info(f"Initialized vectors array with shape: {self.vectors.shape}")
            
            try:
                self.index.reconstruct_n(
                    0, 
                    num_vectors, 
                    self.vectors
                )

            except AssertionError:
                raise RuntimeError(
                    f"Dimension mismatch error. Expected dimension: {dimension}, "
                    f"Index total vectors: {num_vectors}, "
                    f"Index code size: {code_size}. "
                    "The index might be corrupted or incompatible."
                )
            except AttributeError:
                raise RuntimeError(
                    "The loaded index doesn't support vector reconstruction."
                    "Make sure you're using an appropriate index type."
                )

            except Exception as e:
                raise RuntimeError(f"Error extracting vectors: {str(e)}")
                
            return self
        

    @memory(print_report=True)
    @timer(print_time=True)
    def reduce_dimensionality(
        self,
        method: str = "umap",
        pca_components: int = 50,
        final_components: int = 3,
        random_state: int = 42
    ):
        """
        Reduce dimensionality of the extracted vectors with dynamic progress tracking.
        
        Parameters
        ----------
        method : {'pca', 'umap', 'pca_umap'}, optional
            The method to use for dimensionality reduction, by default 'umap'.

            - pca : Principal Component Analysis (PCA) is a linear dimensionality reduction technique 
            that is commonly used to reduce the dimensionality of high-dimensional data. 
            It identifies the directions (principal components) in which the data varies the most 
            and projects the data onto these components, resulting in a lower-dimensional representation.

            - umap : Uniform Manifold Approximation and Projection (UMAP) is a non-linear dimensionality 
            reduction technique that is particularly well-suited for visualizing high-dimensional 
            data in lower-dimensional space. It preserves both local and global structure of the data 
            by constructing a low-dimensional representation that captures the underlying manifold 
            structure of the data.

            - pca_umap : PCA followed by UMAP is a two-step dimensionality reduction technique. 
            First, PCA is applied to reduce the dimensionality of the data. 
            Then, UMAP is applied to further reduce the dimensionality and capture the 
            non-linear structure of the data. This combination can be effective in 
            preserving both global and local structure of the data.
        
        pca_components : int, optional
            Number of components for PCA (used in 'pca' and 'pca_umap'), by default 50.
        
        final_components : int, optional
            Final number of components (3 for 3D visualization), by default 3.
        
        random_state : int, optional
            Random state for reproducibility, by default 42.
        
        Returns
        -------
        self : EmbeddingVisualizer
            The instance itself, allowing for method chaining.
        
        Raises
        ------
        ValueError
            If vectors have not been extracted yet or if an invalid method is specified.
        """
        if self.vectors is None:
            raise ValueError("Vectors not extracted. Call extract_vectors() first.")

        if method == "pca":
            pca = IncrementalPCA(
                n_components=final_components, 
                batch_size=100
            )

            batches = np.array_split(
                self.vectors, 
                max(1, len(self.vectors) // 100)
            )
            
            with tqdm(total=len(batches), desc="PCA") as pbar:
                for batch in batches:
                    pca.partial_fit(batch)
                    pbar.update(1)
            
            self.reduced_vectors = pca.transform(self.vectors)

        elif method == "umap":
            tqdm_out = TqdmToLogger()
            old_stdout = sys.stdout
            sys.stdout = tqdm_out
            
            umap = UMAP(
                n_components=final_components, 
                random_state=random_state, 
                verbose=True
            )
            self.reduced_vectors = umap.fit_transform(self.vectors)
            
            sys.stdout = old_stdout
            tqdm_out.pbar.close()

        elif method == "pca_umap":
            # First, perform PCA
            pca = IncrementalPCA(
                n_components=pca_components, 
                batch_size=100
            )

            batches = np.array_split(
                self.vectors, max(1, len(self.vectors) // 100)
            )
            
            with tqdm(total=len(batches), desc="PCA") as pbar:
                for batch in batches:
                    pca.partial_fit(batch)
                    pbar.update(1)
            
            pca_result = pca.transform(self.vectors)
            
            # Then, perform UMAP
            tqdm_out = TqdmToLogger()
            old_stdout = sys.stdout
            sys.stdout = tqdm_out
            
            umap = UMAP(
                n_components=final_components, 
                random_state=random_state, 
                verbose=True
            )

            self.reduced_vectors = umap.fit_transform(pca_result)
            
            sys.stdout = old_stdout
            tqdm_out.pbar.close()

        else:
            raise ValueError("Invalid method. Choose 'pca', 'umap', or 'pca_umap'.")

        return self


    @memory(print_report=True)
    @timer(print_time=True)
    def create_plot(
        self, 
        title: str = "3D Visualization of Embeddings", 
        point_size: int = 3
    ) -> go.Figure:
        """
        Generate a 3D scatter plot of the reduced vectors with labels.

        Parameters
        ----------
        title : str, optional
            The title of the plot (default is '3D Visualization of Embeddings').

        point_size : int, optional
            The size of the markers in the scatter plot (default is 3).

        Returns
        -------
        go.Figure
            The generated 3D scatter plot.

        Raises
        ------
        ValueError
            If vectors have not been reduced yet.

        Notes
        -----
        This method requires the `plotly` library to be installed.

        Examples
        --------
        >>> visualizer = EmbeddingsVisualizer()
        >>> plot = visualizer.create_plot(title='My Embeddings', point_size=5)
        >>> plot.show()
        """
        if self.reduced_vectors is None:
            raise ValueError("Dimensionality not reduced. Call reduce_dimensionality() first.")

        fig = go.Figure()

        # Start with an empty scatter plot
        scatter = go.Scatter3d(
            x=[], y=[], z=[],
            mode="markers",
            text=[],
            textposition="top center",
            hovertemplate="%{text}",
            hoverinfo="text",
            marker=dict(
                size=point_size,
                color=[],
                colorscale="Viridis",
                opacity=0.8
            )
        )

        fig.add_trace(scatter)

        # Update layout for full-window size and remove axes
        fig.update_layout(
            title=dict(
                text=title,
                y=0.95,
                x=0.5,
                xanchor="center",
                yanchor="top"
            ),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=50, b=50),  # Added some bottom margin for the slider
            autosize=True
        )

        # Add slider
        steps = self._create_slider_steps()
        sliders = [
            dict(
                active=0,
                currentvalue={
                    "prefix": "Number of embeddings: ", 
                    "xanchor": "center"
                },
                pad={
                    "b": 10, 
                    "t": 10
                },
                len=0.9,  # 90% of the plot width
                x=0.05,   # Start at 5% from the left
                xanchor="left",
                y=0,
                yanchor="bottom",
                steps=steps
            )
        ]
        
        fig.update_layout(
            sliders=sliders
        )

        return fig


    def _create_slider_steps(
        self, 
        num_steps: int = 100
    ) -> list:
        """
        Create steps for the slider.

        Parameters:
        ----------
        num_steps : int, optional
            The number of steps to create for the slider. Default is 100.

        Returns:
        -------
        list
            A list of dictionaries representing the steps for the slider.

        Notes:
        ------
        This method creates steps for a slider that can be used to update the visualization of embeddings.

        The steps are created based on the number of vectors in the `reduced_vectors` attribute of the class.
        The `num_steps` parameter determines the number of steps to create. The default value is 100.

        Each step is represented by a dictionary with the following keys:
        - 'method': The update method to be called when the step is selected.
        - 'args': The arguments to be passed to the update method.
        - 'label': The label to be displayed for the step.

        The 'method' key should be set to "update" and the 'args' key should be a list of two dictionaries.
        The first dictionary contains the updated values for the x, y, z, text, and marker.color attributes of the visualization.
        The second dictionary contains the updated title for the visualization.

        The 'label' key should be set to a string representation of the step index.

        Example:
        --------
        >>> visualizer = EmbeddingsVisualizer()
        >>> steps = visualizer._create_slider_steps(num_steps=50)
        >>> print(steps)
        [{'method': 'update', 'args': [{'x': [array([1, 2, 3])], 'y': [array([4, 5, 6])], 'z': [array([7, 8, 9])], 'text': [['label1', 'label2', 'label3']], 'marker.color': [array([7, 8, 9])]}, {'title': 'Embeddings (showing 0 out of 10)'}], 'label': '0'}, ...]
        """
        num_vectors = len(
            self.reduced_vectors
        )

        step_size = max(1, num_vectors // num_steps)
        steps = []


        def _insert_line_breaks(
            text: str, 
            interval: int
        ) -> str:
            """
            Insert line breaks into the text at the specified interval.

            Parameters
            ----------
            text : str
                The input text.

            interval : int
                The interval at which line breaks should be inserted.

            Returns
            -------
            str
                The text with line breaks inserted.

            Examples
            --------
            >>> _insert_line_breaks("Hello, world!", 5)
            'Hello,<br> world!'
            >>> _insert_line_breaks("Lorem ipsum dolor sit amet, consectetur adipiscing elit.", 10)
            'Lorem ipsu<br>m dolor si<br>t amet, co<br>nsectetur<br> adipiscing<br> elit.'
            """
            return "<br>".join(text[i:i+interval] for i in range(0, len(text), interval))


        for i in tqdm(range(0, num_vectors + 1, step_size)):
            # Add HTML line breaks in the text for better formatting
            hover_texts = [_insert_line_breaks(label, 100) for label in self.labels[:i]]
            
            step = dict(
                method="update",
                args=[
                    {
                        "x": [self.reduced_vectors[:i, 0]],
                        "y": [self.reduced_vectors[:i, 1]],
                        "z": [self.reduced_vectors[:i, 2]],
                        "text": [hover_texts],
                        "marker.color": [self.reduced_vectors[:i, 2]]
                    },
                    {
                        "title": f"Embeddings (showing {i} out of {num_vectors})"
                    }
                ],
                label=str(i)
            )
            steps.append(step)

        return steps


    def visualize(
        self, 
        method: str = "tsne", 
        pca_components: int = 50, 
        final_components: int = 3,
        random_state: int = 42,
        title: str = "3D Visualization of Embeddings",
        point_size: int = 3,
        save_html: bool = False,
        html_file_name: str = "embedding_visualization.html"
    ):
        """
        Full pipeline: load index, extract vectors, reduce dimensionality, and visualize.

        Parameters
        ----------
        method : str, optional
            The dimensionality reduction method to use. Default is 'tsne'.
        
        pca_components : int, optional
            The number of components to keep when using PCA for dimensionality reduction. Default is 50.
        
        final_components : int, optional
            The number of final components to visualize. Default is 3.
        
        random_state : int, optional
            The random state for reproducibility. Default is 42.
        
        title : str, optional
            The title of the visualization plot. Default is '3D Visualization of Embeddings'.
        
        point_size : int, optional
            The size of the points in the visualization plot. Default is 3.
        
        save_html : bool, optional
            Whether to save the visualization as an HTML file. Default is False.
        
        html_file_name : str, optional
            The name of the HTML file to save. Default is 'embedding_visualization.html'.

        Returns
        -------
        None

        Raises
        ------
        None

        Examples
        --------
        >>> visualizer = EmbeddingsVisualizer()
        >>> visualizer.visualize(method='tsne', pca_components=50, final_components=3, random_state=42)
        """
        self.load_index()
        self.load_dataset()
        self.extract_vectors()
        self.reduce_dimensionality(
            method=method,
            pca_components=pca_components,
            final_components=final_components,
            random_state=random_state
        )

        fig = self.create_plot(
            title=title,
            point_size=point_size
        )

        if save_html:
            pio.write_html(
                fig, 
                file=html_file_name,
                auto_open=False
            )

            logger.info(f"Visualization saved as {html_file_name}")

       