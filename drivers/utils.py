import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import pickle

def format_transition_matrix(matrix: jnp.ndarray) -> str:
    formatted = "\n".join(["\t" + " ".join(f"{val:.4f}" for val in row) for row in matrix])
    return f"Transition Matrix:\n{formatted}"

def save_model(model, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)  

def load_model(save_path: str):
    with open(save_path, "rb") as f:
        model = pickle.load(f)
    return model


def plot_hmm_diagnostics(model, save_path: str | None = None):
    residuals = np.asarray(model.state_results.pseudo_residuals).ravel()
    residuals = residuals[np.isfinite(residuals)]

    sns.set_theme(style="whitegrid", context="notebook")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ll = np.asarray(model.ll_fits)
    iterations = np.arange(1, len(ll) + 1)
    sns.lineplot(x=iterations, y=ll, ax=axes[0], marker="o")
    axes[0].set_title("Log-likelihood per iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("log L")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_color(sns.color_palette()[0])
    axes[1].get_lines()[1].set_color(sns.color_palette()[3])
    axes[1].set_title("Normal Q-Q of pseudo-residuals")

    lags = min(40, max(1, len(residuals) // 4))
    plot_acf(residuals, lags=lags, ax=axes[2])
    axes[2].set_title("ACF of pseudo-residuals")
    axes[2].set_ylim(-0.25, 1.05)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    return fig
