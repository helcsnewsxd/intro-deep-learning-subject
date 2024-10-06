import pandas as pd
from typing import List, Tuple

import plotly.express as px

import os
from PIL import Image
from io import BytesIO
from IPython.display import display


class Graphics:
    """
    Class for generating and displaying graphics.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame to be used for plotting.
    target : str
        Target column.
    path : str
        Path to the folder where the images are stored or will be stored.
    histograms : List[Image.Image]
        List of histogram images.
    corr : Image.Image
        Correlation matrix image.

    Public Methods
    --------------
    display_histograms()
        Display histograms.
    display_correlation_matrix()
        Display correlation matrix.
    """

    def __init__(self, df: pd.DataFrame, target: str, path: str = None) -> None:
        """
        Initialize the Graphics class.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be used for plotting.
        target : str
            Target column.
        path : str
            Path to the folder where the images are stored.

        Raises
        ------
        ValueError
            If target column not found in DataFrame.

        Returns
        -------
        None
        """
        if target not in df.columns:
            raise ValueError("Target column not found in DataFrame")

        if path is not None and not os.path.exists(path):
            os.makedirs(path)

        self.df = df
        self.target = target
        self.path = path

        (self.histograms, self.corr), loaded = self._load_plots()
        if not loaded and self.path is not None:
            self._save_plots()

    def _get_single_histogram(
            self, x: str, y: str = None, z: str = None, nbins: int = None, has_box: bool = False) -> Image.Image:
        """
        Get a single histogram.

        Parameters
        ----------
        x : str
            Column to be plotted.
        y : str
            Column to be used for coloring.
        z : str
            Column to be used for shape.
        nbins : int
            Number of bins.
        has_box : bool
            Whether to include box plot.

        Raises
        ------
        ValueError
            If column not found in DataFrame.

        Returns
        -------
        Image.Image
            Histogram image.
        """

        if any([col is not None and col not in self.df.columns for col in [x, y, z]]):
            raise ValueError("Column not found in DataFrame")

        marginal = 'box' if has_box else None
        title = f"Histogram of {x}"

        fig = px.histogram(data_frame=self.df, x=x, color=y, pattern_shape=z,
                           marginal=marginal, nbins=nbins, title=title, text_auto=True)
        fig.update_layout(bargap=0.1, xaxis_title=x, yaxis_title='Count', height=600)

        fig_img = Image.open(BytesIO(fig.to_image(format='png')))
        return fig_img

    def _get_all_histograms(self) -> List[Image.Image]:
        """
        Get all histograms.

        Returns
        -------
        List[Image.Image]
            List of histogram images
        """

        fig_list = []
        for x in self.df.columns:
            are_integers = all(self.df[x].apply(lambda x: x.is_integer()))
            nbins = 30 if not are_integers else None

            has_box = self.df[x].nunique() > 2

            fig = self._get_single_histogram(x=x, y=self.target, nbins=nbins, has_box=has_box)
            fig_list.append(fig)

        return fig_list

    def _get_correlation_matrix(self) -> Image.Image:
        """
        Get correlation matrix.

        Returns
        -------
        Image.Image
            Correlation matrix image.
        """

        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns

        corr = self.df[numeric_columns].corr()
        fig = px.imshow(corr, color_continuous_scale='RdBu', title='Correlation Matrix')
        fig.update_layout(width=800, height=800)

        fig_img = Image.open(BytesIO(fig.to_image(format='png')))
        return fig_img

    def _get_plots_from_folder(self) -> List[Image.Image]:
        """
        Get plots from folder.

        Raises
        ------
        ValueError
            If path not provided
        FileNotFoundError
            If no images found in the folder.

        Returns
        -------
        List[Image.Image]
            List of images.
        """

        if self.path is None:
            raise ValueError("Path not provided")

        images = []
        sorted_files = sorted(os.listdir(self.path))
        for filename in sorted_files:
            if filename.endswith('.png'):
                img = Image.open(os.path.join(self.path, filename))
                img.filename = filename
                images.append(img)

        if len(images) == 0:
            raise FileNotFoundError("No images found in the folder")

        return images

    def _load_plots(self) -> Tuple[Tuple[List[Image.Image], Image.Image], bool]:
        """
        Load plots.

        Returns
        -------
        Tuple[Tuple[List[Image.Image], Image.Image], bool]
            Tuple of images and boolean indicating if images were loaded from folder.
        """

        try:
            plots = self._get_plots_from_folder()
            histograms = [img for img in plots if img.filename.startswith('histogram')]
            corr = [img for img in plots if img.filename.startswith('correlation')][0]
            return (histograms, corr), True
        except (ValueError, FileNotFoundError):
            histograms = self._get_all_histograms()
            corr = self._get_correlation_matrix()
            return (histograms, corr), False

    def _save_plots(self) -> None:
        """
        Save plots.

        Raises
        ------
        ValueError
            If path not provided.

        Returns
        -------
        None
        """
        if self.path is None:
            raise ValueError("Path not provided")

        for i, img in enumerate(self.histograms):
            i = str(i).zfill(2)
            img.save(os.path.join(self.path, f'histogram_{i}.png'))

        self.corr.save(os.path.join(self.path, 'correlation.png'))

    def display_histograms(self) -> None:
        """
        Display histograms.

        Returns
        -------
        None
        """

        for img in self.histograms:
            display(img)

    def display_correlation_matrix(self) -> None:
        """
        Display correlation matrix.

        Returns
        -------
        None
        """

        display(self.corr)
