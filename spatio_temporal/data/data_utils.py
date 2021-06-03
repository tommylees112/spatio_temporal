import pandas as pd
from pandas.tseries.frequencies import infer_freq
import xarray as xr
import numpy as np
from numba import njit, prange
from typing import Iterator, List, Tuple, Optional, Union, Any, Dict, DefaultDict
from torch import Tensor
from collections import defaultdict
from spatio_temporal.config import Config
from spatio_temporal.data.normalizer import Normalizer
from tqdm import tqdm


def encode_sample_str_as_int(
    ds: xr.Dataset, sample_str: str
) -> Tuple[xr.Dataset, Dict[int, str]]:
    sample_mapping = dict(enumerate(ds[sample_str].values))
    inverse_mapping = dict((v, k) for k, v in sample_mapping.items())

    ds[sample_str] = [inverse_mapping[v] for v in ds[sample_str].values]

    return ds, sample_mapping


def interpolate_missing_values_in_time(ds: xr.Dataset) -> xr.Dataset:
    #  build an example timeseries from spatial dimensions (latlon or pixel or region)
    #  to then infer the frequency of the timeseries
    df = pd.DataFrame(np.ones(ds.time.values.shape), index=ds.time.values)
    freq = _check_no_missing_times_in_time_series(df)

    #  ensure that all timesteps present
    resample = ds.resample(time=freq).reduce(np.mean)
    #  interpolate nan values in time
    full = resample.interpolate_na(dim="time", method="linear")

    return full


def _alternative_inf_freq(times: Iterator[pd.Timestamp], method="mode") -> pd.Timedelta:
    # https://stackoverflow.com/a/31518059/9940782
    # taking difference of the timeindex and use the mode (or smallest difference) as the freq
    diff = (pd.Series(times[1:]) - pd.Series(times[:-1])).value_counts()

    if method == "mode":
        # the mode can be considered as frequency
        result = diff.index[0]  # output: Timedelta('0 days 01:00:00')
    elif method == "min":
        # or maybe the smallest difference
        result = diff.index.min()  # output: Timedelta('0 days 01:00:00')
    else:
        assert False, "only two possible methods for inferring frequency"

    return result


def _infer_frequency(times: Iterator[pd.Timestamp]) -> Union[str, pd.Timedelta]:
    inf_freq = pd.infer_freq(times)
    if inf_freq is None:
        inf_freq = _alternative_inf_freq(times)
        #  hardcode the monthly timedelta
        if inf_freq == pd.Timedelta("31 days 00:00:00"):
            inf_freq = "M"
    return inf_freq


def _check_no_missing_times_in_time_series(df) -> Union[str, pd.Timedelta]:
    assert (
        df.index.dtype == "datetime64[ns]"
    ), "Need the time index to be of type: datetime64[ns]"
    min_timestamp = df.index.min()
    max_timestamp = df.index.max()
    inf_freq = _infer_frequency(df.index)

    #  inf_data = pd.date_range(start=min_timestamp, end=max_timestamp, freq=inf_freq)
    missing_timesteps = list(
        pd.date_range(start=min_timestamp, end=max_timestamp, freq=inf_freq).difference(
            df.index
        )
    )

    assert missing_timesteps == [], f"Missing data: {missing_timesteps}"

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


def unstack_xarray(
    ds: xr.Dataset, sample_str: str = "pixel", latlon: bool = True
) -> xr.Dataset:
    # lat_lon: str -> [lat, lon]: List[float]
    ll = np.array(
        [
            np.array([float(s) for s in sample.split("_")])
            for sample in ds[sample_str].values
        ]
    )
    lats = ll[:, 0]
    lons = ll[:, 1]

    #  create xarray with unstacked sample_str dimension
    df = ds.to_dataframe()

    if latlon:
        df["lat"] = lats
        df["lon"] = lons
        ds = df.reset_index().set_index(["time", "lat", "lon"]).to_xarray()
    else:
        df = df.reset_index().rename({sample_str: "index"}, axis=1)
        df[sample_str] = df["index"].str.split("_").str[0].astype(float)
        ds = df.set_index(["time", sample_str]).to_xarray()

    return ds


# @njit
def validate_samples(
    x_d: List[np.ndarray],
    x_s: List[np.ndarray],
    x_f: List[np.ndarray],
    y: List[np.ndarray],
    seq_length: int,
    forecast_horizon: int,
    mode: str,
) -> np.ndarray:
    n_samples = len(x_d)
    flag = np.ones(n_samples)

    # if any condition met then go to next iteration of loop
    for current_index in prange(n_samples):
        target_index = current_index + forecast_horizon

        # 1. not enough history (seq_length > history)
        if target_index < seq_length:
            flag[current_index] = 0
            continue

        #  5. not enough data for forecast horizon
        if n_samples < (target_index + forecast_horizon):
            flag[current_index] = 0
            continue
        #  min_x > required seq_len (allow for zero based indexing)
        if target_index < (seq_length + forecast_horizon):
            flag[current_index] = 0
            continue

        #  NOTE: indexing here needs to be the same as in dataloader.__getitem__
        #  2. NaN in the dynamic inputs
        end_input_idx_plus_1 = current_index + 1
        start_input_idx = end_input_idx_plus_1 - seq_length
        _x_d = x_d[start_input_idx:end_input_idx_plus_1]

        if np.any(np.isnan(_x_d)):
            flag[current_index] = 0
            continue

        #  NOTE: indexing here needs to be the same as in dataloader.__getitem__
        #  3. NaN in the outputs (only for training period)
        if mode != "test":
            # _y = y[target_index]
            _y = y[start_input_idx: end_input_idx_plus_1]

            # if np.isnan(_y):
            if np.prod(np.array(_y.shape)) > 0 and np.all(np.isnan(_y)):
                flag[current_index] = 0
                continue

        # 4. any NaN in the static features makes the target_index invalid
        if x_s is not None:
            _x_s = x_s
            if np.any(np.isnan(_x_s)):
                flag[current_index] = 0
                continue

        # 6. any NaN in the forecast data
        #  NOTE: indexing here needs to be the same as in dataloader.__getitem__
        if x_f is not None:
            start_input_idx = end_input_idx_plus_1 - seq_length
            _x_d = x_d[start_input_idx : target_index + 1]

            if np.any(np.isnan(_x_d)):
                flag[current_index] = 0
                continue

    return flag


def _reshape(array: np.ndarray) -> np.ndarray:
    return array if array.ndim > 1 else array.reshape(-1, 1)


def load_all_data_from_dl_into_memory(dl: Any) -> Dict[str, np.ndarray]:
    out: DefaultDict[str, List] = defaultdict(list)
    pbar1 = tqdm(dl, desc="Extracting data from DataLoader")
    for data in pbar1:
        # TODO: don't do this on GPU ..?
        out["x_d"].append(data["x_d"].detach().cpu().numpy())
        out["y"].append(data["y"].detach().cpu().numpy())
        out["time"].append(data["meta"]["target_time"].detach().cpu().numpy())
        out["index"].append(data["meta"]["index"].detach().cpu().numpy())

    return_dict: Dict[str, np.ndarray] = {}
    pbar2 = tqdm(out.keys(), desc="Concatenating data & writing to return_dict")
    for key in pbar2:
        # concatenate over batch dimension (dimension = 0)
        var_ = np.concatenate(out[key])
        var_ = var_.squeeze() if var_.ndim == 3 else var_
        var_ = _reshape(var_)
        return_dict[key] = var_

    return return_dict


def train_test_split(ds: xr.Dataset, cfg: Config, subset: str) -> xr.Dataset:
    # TODO: define sample strategy in space as well as time.
    #  i.e. define train/test basins (PUB)
    input_variables = [] if cfg.input_variables is None else cfg.input_variables
    forecast_variables = (
        [] if cfg.forecast_variables is None else cfg.forecast_variables
    )
    #  ensure that ds is sorted by time
    ds = ds.sortby("time")

    if subset == "train":
        ds = ds[input_variables + [cfg.target_variable] + forecast_variables].sel(
            time=slice(cfg.train_start_date, cfg.train_end_date)
        )
    elif subset == "validation":
        ds = ds[input_variables + [cfg.target_variable] + forecast_variables].sel(
            time=slice(cfg.validation_start_date, cfg.validation_end_date)
        )
    elif subset == "test":
        ds = ds[input_variables + [cfg.target_variable] + forecast_variables].sel(
            time=slice(cfg.test_start_date, cfg.test_end_date)
        )
    else:
        assert False, f"No such subset ({subset}) can be supplied"

    assert 0 not in [
        v for v in ds.dims.values()
    ], f"{subset} Period returns NO samples {ds}"

    return ds


def encode_doys(
    doys: Union[int, List[int]], start_doy: int = 1, end_doy: int = 366
) -> Tuple[List[float], List[float]]:
    """
    encode (list of) date(s)/doy(s) to cyclic sine/cosine values
    int is assumed to represent a day of year
    
    it is possible to change the encoding period by passing `start_doy` or
    `end_doy` to the function
    (e.g. if you want to have cyclic values for the vegetation period only)
    returns two lists, one with sine-encoded and one with cosine-encoded doys
    """
    if not isinstance(doys, list):
        doys = [doys]

    doys_sin = []
    doys_cos = []
    for doy in doys:
        if doy > 366 or doy < 1:
            raise ValueError(f'Invalid date "{doy}"')

        doys_sin.append(
            np.sin(2 * np.pi * (doy - start_doy) / (end_doy - start_doy + 1))
        )
        doys_cos.append(
            np.cos(2 * np.pi * (doy - start_doy) / (end_doy - start_doy + 1))
        )

    return doys_sin, doys_cos


def initialize_normalizer(
    ds: xr.Dataset,
    cfg: Config,
    collapse_dims: Optional[List[str]] = ["time"],
    normalizer: Optional[Normalizer] = None,
) -> Normalizer:
    normalizer = Normalizer(fit_ds=ds, collapse_dims=collapse_dims)

    #  Manually set the mean_ / std_ (defined in cfg)
    if cfg.constant_mean is not None:
        # create a mean value for each variable
        for variable in [k for k in cfg.constant_mean.keys()]:
            normalizer.update_mean_with_constant(
                variable=variable, mean_value=cfg.constant_mean[variable]
            )
    if cfg.constant_std is not None:
        for variable in [k for k in cfg.constant_std.keys()]:
            # create a std value for each variable
            normalizer.update_std_with_constant(
                variable=variable, std_value=cfg.constant_std[variable]
            )

    return normalizer


def add_doy_encoding_as_feature_to_dataset(
    ds: xr.Dataset, inputs: List[str], target: str
) -> Tuple[xr.Dataset, List[str]]:
    #  create sin/cosin of doy
    dts = pd.to_datetime(ds.time.values)
    sin_doy, cos_doy = encode_doys([d.dayofyear for d in dts])

    # store as xr.DataArray objects
    sin_doy_xr: xr.DataArray = xr.ones_like(ds[target]) * np.tile(
        sin_doy, len(ds.sample.values)
    ).reshape(-1, len(ds.sample.values))
    sin_doy_xr = sin_doy_xr.rename("sin_doy")

    cos_doy_xr: xr.DataArray = xr.ones_like(ds[target]) * np.tile(
        cos_doy, len(ds.sample.values)
    ).reshape(-1, len(ds.sample.values))
    cos_doy_xr = cos_doy_xr.rename("cos_doy")

    # update xr.Dataset
    ds = xr.merge([ds, sin_doy_xr, cos_doy_xr])
    # update inputs
    inputs = inputs + ["sin_doy", "cos_doy"]

    return ds, inputs
