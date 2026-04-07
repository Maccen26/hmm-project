import jax.numpy as jnp
import pandas as pd
from src.data import load_and_aggregate_data
def load_y_data(no_of_days: int | None = None) -> jnp.ndarray:
    """
    Loads the y data from the csv file and returns it as a jnp array. 
    If no_of_days is not None, it returns only the first no_of_days of data. 
    """ 
    df = load_and_aggregate_data(no_of_days=no_of_days)
    y_data = jnp.array(df["CO2C"].values)
    return jnp.asarray(y_data)


def make_lag_matrix(ys: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    Build (T, k) lag matrix where row t = [y_{t-1}, ..., y_{t-k}].
    Rows t < k are zero-padded (valid lags don't exist yet).
    """
    T = len(ys)
    cols = [jnp.concatenate([jnp.zeros(lag), ys[:T - lag]]) for lag in range(1, k + 1)]
    return jnp.stack(cols, axis=1)  # (T, k)