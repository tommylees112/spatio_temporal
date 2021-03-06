import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
import xarray as xr
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from spatio_temporal.data.data_utils import (
    _check_no_missing_times_in_time_series,
    _stack_xarray,
    validate_samples,
    encode_doys,
)
from spatio_temporal.data.normalizer import Normalizer
from spatio_temporal.config import Config


class XarrayDataset(Dataset):
    """A simple torch Dataset wrapping xr.DataArray

    Args:
        data (xr.Dataset): Data including target and input data
        cfg (Config): config object describing the experiment run

    Attrs:
        target_variable (str): The name of the target variable in `ds`
        input_variables (List[str]): name of the input variables used for ds
        seq_length (int): the number of input timesteps to include (input window length)
        mode (str): {"train", "test", "validation}
        horizon (int, optional): Forecast horizon. Defaults to 1.
        pixel_dims (List[str], optional): Dimensions describing the spatial sample. Defaults to ["lat", "lon"].
        time_dim (str, optional): Dimension describing time. Defaults to "time".
        autoregressive (bool, optional): Whether to include the target as input (shifted +1). Defaults to False.
    """

    def __init__(
        self,
        data: xr.Dataset,
        cfg: Config,
        mode: str,
        normalizer: Optional[Normalizer] = None,
        DEBUG: bool = False,
    ):
        self.cfg = cfg
        # stack pixel_dims to one dimension ("sample")
        stacked, sample = _stack_xarray(data, self.cfg.pixel_dims)

        self.mode = mode
        self.is_train = self.mode == "train"
        self.sample: xr.Dataset = sample
        self.batch_dim: str = "sample"
        self.seq_length: int = self.cfg.seq_length
        self.inputs: Optional[
            List[str]
        ] = self.cfg.input_variables if self.cfg.input_variables is not None else []
        self.target: str = self.cfg.target_variable
        self.device = self.cfg.device
        self.DEBUG = DEBUG

        # TODO: allow forecast variables
        self.forecast_variables = self.cfg.forecast_variables
        if self.forecast_variables is not None:
            assert (
                self.cfg.target_variable not in self.cfg.forecast_variables
            ), "Cannot include target as a forecast variable (leakage)"

        ds: xr.Dataset = stacked

        # TODO: allow static inputs
        #  TODO: replace "x_s" with "x_one_hot"
        self.static_inputs = cfg.static_inputs
        if self.static_inputs == "embedding":
            self.df_static = pd.DataFrame(
                torch.nn.functional.one_hot(torch.arange(ds.sample.size)).numpy(),
                columns=ds.sample.values,
                index=ds.sample.values,
            )
            self.static_inputs = self.df_static.columns

        #  TODO: make normalizer optional (e.g. for Runoff data)
        #  TODO: normalize only specific variables, e.g. inputs not outputs
        ds = self.run_normalization(ds=ds, normalizer=normalizer)
        if self.DEBUG:
            # save the stacked dataset to memory to check dataloading
            self.ds = ds

        # information for building models
        self.horizon = cfg.horizon
        self.output_size = 1

        # add autoregressive variable
        if cfg.autoregressive:
            assert (
                self.horizon > 0
            ), "Cannot include autoregressive information if simulating current timestep. Use a horizon of 1 instead."
            auto = ds[self.target].rename("autoregressive")
            self.inputs = self.inputs + ["autoregressive"]
            ds = xr.merge([ds, auto])

        # add doy encoding
        if cfg.encode_doys:
            dts = pd.to_datetime(ds.time.values)
            sin_doy, cos_doy = encode_doys([d.dayofyear for d in dts])
            self.inputs = self.inputs + ["sin_doy", "cos_doy"]
            sin_doy_xr = xr.ones_like(ds[self.target]) * np.tile(
                sin_doy, len(sample.sample.values)
            ).reshape(-1, len(sample.sample.values))
            sin_doy_xr = sin_doy_xr.rename("sin_doy")
            cos_doy_xr = xr.ones_like(ds[self.target]) * np.tile(
                cos_doy, len(sample.sample.values)
            ).reshape(-1, len(sample.sample.values))
            cos_doy_xr = cos_doy_xr.rename("cos_doy")
            ds = xr.merge([ds, sin_doy_xr, cos_doy_xr])

        #  ---- DEFINE INPUT SIZES ----
        # (used to create the models later)
        self.dynamic_input_size = len(self.inputs)
        self.static_input_size = (
            len(self.static_inputs) if self.static_inputs is not None else 0
        )
        self.forecast_input_size = (
            len(self.forecast_variables) if self.forecast_variables is not None else 0
        )

        # init dictionaries to store the RAW pixel timeseries
        self.x_d: Dict[str, np.ndarray] = {}
        self.y: Dict[str, np.ndarray] = {}
        self.times: Dict[str, Optional[np.ndarray]] = {}
        self.x_s: Dict[str, Optional[np.ndarray]] = {}
        self.x_f: Dict[str, Optional[np.ndarray]] = {}

        # 1. Check for missing data
        # 2. Store int -> data index
        # 3. Store y, x_d, x_s
        self.create_lookup_table(ds)

    def run_normalization(
        self,
        ds,
        collapse_dims: List[str] = ["time"],
        normalizer: Optional[Normalizer] = None,
    ) -> xr.Dataset:
        if self.mode == "train":
            self.normalizer = Normalizer(fit_ds=ds, collapse_dims=collapse_dims)

            #  Manually set the mean_ / std_ (defined in cfg)
            if self.cfg.constant_mean is not None:
                for variable in [k for k in self.cfg.constant_mean.keys()]:
                    self.normalizer.update_mean_with_constant(
                        variable=variable, mean_value=self.cfg.constant_mean[variable]
                    )
            if self.cfg.constant_std is not None:
                for variable in [k for k in self.cfg.constant_std.keys()]:
                    self.normalizer.update_std_with_constant(
                        variable=variable, std_value=self.cfg.constant_std[variable]
                    )

            # save the normalizer into the run directory
            pickle.dump(
                self.normalizer, (self.cfg.run_dir / "normalizer.pkl").open("wb")
            )
        else:
            if normalizer is None:
                self.normalizer = pickle.load(
                    (self.cfg.run_dir / "normalizer.pkl").open("rb")
                )
            else:
                self.normalizer = normalizer

        ds = self.normalizer.transform(ds)
        return ds

    def store_data(
        self,
        pixel: str,
        x_d: np.ndarray,
        y: np.ndarray,
        times: Optional[np.ndarray] = None,
        x_s: Optional[np.ndarray] = None,
        x_f: Optional[np.ndarray] = None,
    ):
        self.x_d[pixel] = torch.from_numpy(x_d.astype(np.float32))
        self.y[pixel] = torch.from_numpy(y.astype(np.float32))
        if self.static_inputs is not None:
            self.x_s[pixel] = torch.from_numpy(x_s.astype(np.float32))
        if self.forecast_variables is not None:
            self.x_f[pixel] = torch.from_numpy(x_f.astype(np.float32))

        # store metadata
        if times is not None:
            #  NOTE: this is super inefficient becuase duplicated over each pixel
            #  store as float32 to keep pytorch happy
            time_ = np.array(times) if not isinstance(times, np.ndarray) else times
            self.times[pixel] = torch.from_numpy(np.array(time_).astype(np.float32))

    def create_lookup_table(self, ds):
        pixels_without_samples = []
        lookup: List[Tuple[str, int]] = []

        for pixel in tqdm(ds.sample.values, desc="Loading Data: "):
            df_native = ds.sel(sample=pixel).to_dataframe()
            self.freq = _check_no_missing_times_in_time_series(df_native)

            #  store times as float
            times = df_native.index

            # TODO: Include forecasted variables into dynamic
            if self.forecast_variables is not None:
                x_f = df_native[self.forecast_variables].values
            else:
                x_f = None

            x_d = df_native[self.inputs].values
            y = df_native[self.target].values

            #  TODO: deal with static inputs
            if self.static_inputs is not None:
                # index = pixel; columns = variables
                x_s = self.df_static.loc[pixel, self.static_inputs].values
                #  TODO: check if error in shaping
                # np.tile(x_s, y.size).reshape(y.size, x_s.size)
            else:
                x_s = None

            # checks inputs and outputs for each sequence. valid: flag = 1, invalid: flag = 0
            flag = validate_samples(
                x_d=x_d,
                x_s=x_s,
                y=y,
                x_f=x_f,
                seq_length=self.seq_length,
                forecast_horizon=self.horizon,
                mode=self.mode,
            )

            # store lookups with Pixel_ID and Index for that valid sample
            valid_samples = np.argwhere(flag == 1)
            [lookup.append((pixel, smp)) for smp in valid_samples]

            # STORE DATA if basin has at least ONE valid sample
            if valid_samples.size > 0:
                self.store_data(pixel, x_d=x_d, y=y, x_s=x_s, x_f=x_f, times=times)
            else:
                pixels_without_samples.append(pixel)

        # Lookup table: int -> (pixel, index)
        self.lookup_table: Dict[int, Tuple[str, int]] = {
            i: elem for i, elem in enumerate(lookup)
        }
        self.num_samples = len(self.lookup_table)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        #  get the valid sample information for pixel, current time idx
        pixel, valid_current_time_index = self.lookup_table[idx]
        valid_current_time_index = int(valid_current_time_index)
        data = {}

        target_index = valid_current_time_index + self.horizon

        #  INPUT DATA (add 1 for 0 based indexing)
        #  get the inputs for [start_input_idx : end_input_idx]
        end_input_idx_plus_1 = (valid_current_time_index) + 1
        start_input_idx = end_input_idx_plus_1 - self.cfg.seq_length
        x_d = self.x_d[pixel][start_input_idx:end_input_idx_plus_1]
        if self.DEBUG:
            assert x_d.shape[0] == self.cfg.seq_length

        if self.forecast_variables is not None:
            #  up to and including the target time
            x_f = self.x_f[pixel][start_input_idx : target_index + 1]
            assert x_f.shape[0] == self.cfg.seq_length + self.cfg.horizon
        else:
            x_f = torch.from_numpy(np.array([]))

        #  TARGET DATA
        y = self.y[pixel][target_index].reshape(-1, 1)

        if self.static_inputs is not None:
            # torch.cat((self.x_s[pixel]), dim=-1)
            x_s = self.x_s[pixel]
        else:
            x_s = torch.from_numpy(np.array([]))

        # METADATA
        # store time as float32
        # convert back to timestamp https://stackoverflow.com/a/47562725/9940782
        time = self.times[pixel][target_index]
        tgt_index = torch.from_numpy(np.array([target_index]).reshape(-1))

        # # write output dictionary
        meta = {
            "index": idx,
            "target_time": time,
        }
        data["meta"] = meta

        data["x_d"] = x_d
        data["y"] = y
        data["x_s"] = x_s
        data["x_f"] = x_f

        return data
