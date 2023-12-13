import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ! ----------------------Principal Component Analysis-------------------------

def plot_variance(pca: PCA, width: int = 8, dpi: int = 100) -> np.ndarray:
    """
    Create a pair of subplots to visualize explained variance and cumulative variance
    of Principal Component Analysis (PCA).

    Args:
        pca (PCA): The fitted PCA model.
        width (int): Width of the figure (default is 8).
        dpi (int): Dots per inch for figure resolution (default is 100).

    Returns:
        np.ndarray: An array containing two matplotlib.axes._subplots.AxesSubplot objects.
    Source: https://www.kaggle.com/code/ryanholbrook/principal-component-analysis/tutorial
    """

    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )

    fig.set_figwidth(width)
    fig.set_dpi(dpi)

    return axs