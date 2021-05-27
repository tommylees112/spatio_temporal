import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
import xarray as xr
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any, cast
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from spatio_temporal.data.data_utils import (
    _check_no_missing_times_in_time_series,
    _stack_xarray,
    validate_samples,
    encode_doys,
    initialize_normalizer,
    add_doy_encoding_as_feature_to_dataset,
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
        static_data: Optional[xr.Dataset] = None,
    ):
        self.cfg = cfg

        if self.cfg.static_data_path is not None:
            assert static_data is not None, "Expect static dataset to be provided"

        # stack pixel_dims to one dimension ("sample")
        stacked, sample = _stack_xarray(data, self.cfg.pixel_dims)
        stacked_static: Optional[xr.Dataset]
        if static_data is not None:
            stacked_static, _ = _stack_xarray(static_data, self.cfg.pixel_dims)
        else:
            stacked_static = None

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
                self.cfg.target_variable not in self.cfg.forecast_variables  # type: ignore
            ), "Cannot include target as a forecast variable (leakage)"

        #  -------- data ----------
        #  The data as it is loaded into the models
        ds: xr.Dataset = stacked
        static: Optional[xr.Dataset] = stacked_static

        #  -------- normalization ----------
        if self.cfg.dynamic_normalization:
            #  TODO: make normalizer optional (e.g. for Runoff data)
            #  TODO: normalize only specific variables, e.g. inputs not outputs
            ds = self.run_normalization(ds=ds, normalizer=normalizer)
        else:
            self.normalizer = None
        
        if (static is not None) and (self.cfg.static_normalization):
            static = self.run_static_normalization(ds=static)
        else:
            self.static_normalizer = None

        # TODO: allow static inputs
        #  TODO: replace "x_s" with "x_one_hot"
        self.static_inputs = cfg.static_inputs
        self.df_static: Optional[pd.DataFrame]

        if self.static_inputs == "embedding":
            self.df_static = pd.DataFrame(
                torch.nn.functional.one_hot(torch.arange(ds.sample.size)).numpy(),
                columns=ds.sample.values,
                index=ds.sample.values,
            )
            self.static_inputs = self.df_static.columns
        elif isinstance(self.static_inputs, list):
            assert (
                static is not None
            ), "Expected static data to be passed to Dataset initialisation"
            self.df_static = static.to_dataframe()
        else:
            self.df_static = None

        if self.DEBUG:
            # save the stacked dataset to memory to check dataloading
            self.ds = ds
            self.static = static

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
            ds, self.inputs = add_doy_encoding_as_feature_to_dataset(
                ds, inputs=self.inputs, target=self.target
            )

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
        self.times: Dict[str, np.ndarray] = {}
        self.x_s: Dict[str, Optional[np.ndarray]] = {}
        self.x_f: Dict[str, Optional[np.ndarray]] = {}
        self.target_std: Dict[str, Optional[np.ndarray]] = {}

        # get the std of the target for each pixel
        # only needed for some losses (NSE)
        self._calculate_per_pixel_target_std(ds)

        # 1. Check for missing data
        # 2. Store int -> data index
        # 3. Store y, x_d, x_s
        self.create_lookup_table(ds)

    def _run_normalization(
        self,
        ds: xr.Dataset,
        collapse_dims: Optional[List],
        static: bool = False,
        normalizer: Optional[Normalizer] = None,
    ) -> Tuple[xr.Dataset, Normalizer]:
        filename: str = "static_normalizer.pkl" if static else "normalizer.pkl"
        if self.mode == "train":
            normalizer = initialize_normalizer(
                ds=ds, cfg=self.cfg, collapse_dims=collapse_dims, normalizer=normalizer
            )
            pickle.dump(normalizer, (self.cfg.run_dir / filename).open("wb"))

        else:
            #  Normalizer already fit on the training data & saved to disk
            if normalizer is None:
                normalizer = pickle.load((self.cfg.run_dir / filename).open("rb"))

        ds = normalizer.transform(ds, variables=self.cfg.normalize_variables)

        return ds, normalizer

    def run_static_normalization(
        self, ds: xr.Dataset, normalizer: Optional[Normalizer] = None,
    ) -> xr.Dataset:
        ds, normalizer = self._run_normalization(
            ds=ds, static=True, collapse_dims=None, normalizer=normalizer
        )
        self.static_normalizer = normalizer
        return ds

    def run_normalization(
        self,
        ds: xr.Dataset,
        collapse_dims: Optional[List[str]] = ["time"],
        normalizer: Optional[Normalizer] = None,
    ) -> xr.Dataset:
        ds, normalizer = self._run_normalization(
            ds=ds, static=False, collapse_dims=collapse_dims, normalizer=normalizer
        )
        self.normalizer = normalizer
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
            x_s = cast(np.ndarray, x_s)  #  make mypy happy
            self.x_s[pixel] = torch.from_numpy(x_s.astype(np.float32))  #  type: ignore
        if self.forecast_variables is not None:
            x_f = cast(np.ndarray, x_f)  #  make mypy happy
            self.x_f[pixel] = torch.from_numpy(
                (x_f.astype(np.float32))
            )  #  type: ignore

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

            # Include forecasted variables into dynamic
            #  TODO: pad forecast variables with zero so only including the one forecast var
            if self.forecast_variables is not None:
                x_f = df_native[self.forecast_variables].values
            else:
                x_f = None

            x_d = df_native[self.inputs].values
            y = df_native[self.target].values

            if self.static_inputs is not None:
                # index = pixel; columns = variables
                x_s = self.df_static.loc[pixel, self.static_inputs].values
                #  check if error in shaping
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

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Tensor]]:
        #  get the valid sample information for pixel, current time idx
        pixel, valid_current_time_index = self.lookup_table[idx]
        valid_current_time_index = int(valid_current_time_index)

        target_index = valid_current_time_index + self.horizon

        #  INPUT DATA (add 1 for 0 based indexing)
        #  get the inputs for [start_input_idx : end_input_idx]
        end_input_idx_plus_1 = (valid_current_time_index) + 1
        start_input_idx = end_input_idx_plus_1 - self.cfg.seq_length
        x_d = self.x_d[pixel][start_input_idx:end_input_idx_plus_1]

        assert x_d.shape[0] == self.cfg.seq_length

        if self.forecast_variables is not None:
            #  up to and including the target time
            x_f = self.x_f[pixel][start_input_idx : target_index + 1]  #  type: ignore
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

        # required for NSE loss
        if self.target_std:
            pixel_target_std = self.target_std[pixel]
        else:
            pixel_target_std = torch.from_numpy(np.array([]))

        # METADATA
        # store time as float32
        # convert back to timestamp https://stackoverflow.com/a/47562725/9940782
        time = self.times[pixel][target_index]
        tgt_index = torch.from_numpy(np.array([target_index]).reshape(-1))

        meta = {
            "index": idx,
            "target_time": time,
        }

        # write output dictionary
        data = {}
        data["meta"] = meta
        data["x_d"] = x_d
        data["y"] = y
        data["x_s"] = cast(Any, x_s)  #  make mypy happy
        data["x_f"] = x_f
        data['target_std'] = pixel_target_std

        return data

    def _calculate_per_pixel_target_std(self, ds: xr.Dataset):
        pixels = ds["sample"].values.tolist()
        for pixel in pixels:
            # select pixel target data
            target = ds.sel(sample=pixel)[self.cfg.target_variable].values
            # calculate std for each target
            pixel_target_std = torch.tensor([np.nanstd(target)], dtype=torch.float32)

            self.target_std[pixel] = pixel_target_std
