import pandas as pd
import xarray as xr
import numpy as np
from numba import njit, prange
from typing import List, Tuple, Optional, Union, Any, Dict, DefaultDict
from torch import Tensor
from collections import defaultdict


def _alternative_inf_freq(df, method="mode") -> pd.Timedelta:
    # https://stackoverflow.com/a/31518059/9940782
    # taking difference of the timeindex and use the mode (or smallest difference) as the freq
    diff = (pd.Series(df.index[1:]) - pd.Series(df.index[:-1])).value_counts()

    if method == "mode":
        # the mode can be considered as frequency
        result = diff.index[0]  # output: Timedelta('0 days 01:00:00')
    elif method == "min":
        # or maybe the smallest difference
        result = diff.index.min()  # output: Timedelta('0 days 01:00:00')
    else:
        assert False, "only two possible methods for inferring frequency"

    return result


def _check_no_missing_times_in_time_series(df) -> Union[str, pd.Timedelta]:
    assert (
        df.index.dtype == "datetime64[ns]"
    ), "Need the time index to be of type: datetime64[ns]"
    min_timestamp = df.index.min()
    max_timestamp = df.index.max()
    inf_freq = pd.infer_freq(df.index)
    if inf_freq is None:
        inf_freq = _alternative_inf_freq(df)

    #  inf_data = pd.date_range(start=min_timestamp, end=max_timestamp, freq=inf_freq)
    assert (
        list(
            pd.date_range(
                start=min_timestamp, end=max_timestamp, freq=inf_freq
            ).difference(df.index)
        )
        == []
    ), f"Missing data"

    return inf_freq


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
    x_d: List[np.ndarray],
    x_s: List[np.ndarray],
    y: List[np.ndarray],
    seq_length: int,
    forecast_horizon: int,
) -> np.ndarray:
    n_samples = len(y)
    flag = np.ones(n_samples)

    # if any condition met then go to next iteration of loop
    for target_index in prange(n_samples):
        # 1. not enough history (seq_length > history)
        if target_index < seq_length:
            flag[target_index] = 0
            continue

        #  5. not enough data for forecast horizon
        if n_samples < (target_index + forecast_horizon + 1):
            flag[target_index] = 0
            continue
        #  min_x > required seq_len (allow for zero based indexing)
        if target_index < (seq_length + forecast_horizon - 1):
            flag[target_index] = 0
            continue

        #  NOTE: indexing here needs to be the same as in dataloader.__getitem__
        #  2. NaN in the dynamic inputs
        _x_d = x_d[
            ((target_index - seq_length - forecast_horizon) + 1) : (
                target_index - forecast_horizon
            )
            + 1
        ]
        if np.any(np.isnan(_x_d)):
            flag[target_index] = 0
            continue

        #  NOTE: indexing here needs to be the same as in dataloader.__getitem__
        #  3. NaN in the outputs (only for training period)
        if y is not None:
            _y = y[target_index : (target_index + forecast_horizon)]

            if np.any(np.isnan(y)):
                flag[target_index] = 0
                continue

        # 4. any NaN in the static features makes the target_index invalid
        if x_s is not None:
            _x_s = x_s[target_index]
            if np.any(np.isnan(_x_s)):
                flag[target_index] = 0
                continue

    return flag


def _reshape(array: np.ndarray) -> np.ndarray:
    return array if array.ndim > 1 else array.reshape(-1, 1)


def load_all_data_from_dl_into_memory(dl: Any) -> Tuple[np.ndarray, ...]:
    out: DefaultDict[List] = defaultdict(list)
    for data in dl:
        # TODO: don't do this on GPU ..?
        out["x_d"].append(data["x_d"].detach().cpu().numpy())
        out["y"].append(data["y"].detach().cpu().numpy())
        out["time"].append(data["meta"]["target_time"].detach().cpu().numpy())
        out["index"].append(data["meta"]["index"].detach().cpu().numpy())

    return_dict: Dict[str, np.ndarray] = {}
    for key in out.keys():
        # concatenate over batch dimension (dimension = 0)
        var_ = np.concatenate(out[key])
        var_ = var_.squeeze() if var_.ndim == 3 else var_
        var_ = _reshape(var_)
        return_dict[key] = var_

    return return_dict
