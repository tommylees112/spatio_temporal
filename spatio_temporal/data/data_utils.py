import pandas as pd
import xarray as xr
import numpy as np
from numba import njit, prange
from typing import List, Tuple, Optional
from sklearn_xarray import wrap
from sklearn.preprocessing import StandardScaler


def normalise_dataset(ds: xr.Dataset, scaler: Optional[StandardScaler] = None):
    if scaler is None:
        # 'train' mode
        scaler = StandardScaler()
        scaler = wrap(scaler).fit(ds).estimator

    norm_ds = wrap(scaler).transform(ds)
    assert False
    return scaler, norm_ds


def _check_no_missing_times_in_time_series(df):
    min_timestamp = df.index.min()
    max_timestamp = df.index.max()
    inf_freq = pd.infer_freq(df.index)
    assert (
        list(
            pd.date_range(
                start=min_timestamp, end=max_timestamp, freq=inf_freq
            ).difference(df.index)
        )
        == []
    ), f"Missing data"


def _stack_xarray(
    ds: xr.Dataset, spatial_coords: List[str]
) -> Tuple[xr.Dataset, xr.Dataset]:
    #  stack values
    stacked = ds.stack(sample=spatial_coords)
    samples = stacked.sample
    pixel_strs = [f"{ll[0]}_{ll[-1]}" for ll in samples.values]
    stacked["sample"] = pixel_strs

    samples = samples.to_dataset(name="pixel")
    samples = xr.DataArray(
        pixel_strs, dims=["sample"], coords={"sample": samples.sample}
    )
    return stacked, samples


@njit
def validate_samples(
    x_d: List[np.ndarray], x_s: List[np.ndarray], y: List[np.ndarray], seq_length: int
) -> np.ndarray:
    n_samples = len(y)
    flag = np.ones(n_samples)

    # if any condition met then go to next iteration of loop
    for sample_idx in prange(n_samples):
        # 1. not enough history (seq_length > history)
        if sample_idx < seq_length:
            flag[sample_idx] = 0
            continue

        #  2. NaN in the dynamic inputs
        _x_d = x_d[sample_idx - seq_length + 1 : sample_idx + 1]
        if np.any(np.isnan(_x_d)):
            flag[sample_idx] = 0
            continue

        #  3. NaN in the outputs (only for training period)
        if y is not None:
            _y = y[sample_idx - seq_length + 1 : sample_idx + 1]
            if np.any(np.isnan(y)):
                flag[sample_idx] = 0
                continue

        # any NaN in the static features makes the sample_idx invalid
        if x_s is not None:
            _x_s = x_s[sample_idx]
            if np.any(np.isnan(_x_s)):
                flag[sample_idx] = 0
                continue

    return flag
