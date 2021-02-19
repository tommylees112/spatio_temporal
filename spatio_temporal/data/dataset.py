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
from spatio_temporal.data.data_utils import (
    _check_no_missing_times_in_time_series,
    _stack_xarray,
    validate_samples,
)
from spatio_temporal.data.normaliser import Normalizer
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
        self.inputs: List[str] = self.cfg.input_variables
        self.target: str = self.cfg.target_variable
        self.device = self.cfg.device
        self.DEBUG = DEBUG

        # TODO: allow static inputs
        self.static_inputs = cfg.static_inputs

        # TODO: allow forecast variables
        self.forecast_variables = self.cfg.forecast_variables

        ds: xr.Dataset = stacked
        ds = self.run_normalisation(ds, normalizer=normalizer)
        if self.DEBUG:
            # save the stacked dataset to memory to check dataloading
            self.ds = ds

        # information for building models
        self.horizon = cfg.horizon
        self.output_size = self.horizon if self.horizon > 0 else 1

        # add autoregressive variable
        if cfg.autoregressive:
            assert (
                self.horizon > 0
            ), "Cannot include autoregressive information if simulating current timestep. Use a horizon of 1 instead."
            auto = stacked[self.target].rename("autoregressive")
            self.inputs = self.inputs + ["autoregressive"]
            ds = xr.merge([ds, auto])

        self.input_size = len(self.inputs)

        # init dictionaries to store the RAW pixel timeseries
        self.x_d: Dict[str, np.ndarray] = {}
        self.y: Dict[str, np.ndarray] = {}
        self.times: Dict[str, Optional[np.ndarray]] = {}
        self.x_s: Dict[str, Optional[np.ndarray]] = {}

        # 1. Check for missing data
        # 2. Store int -> data index
        # 3. Store y, x_d, x_s
        self.create_lookup_table(ds)

    def run_normalisation(
        self,
        ds,
        collapse_dims: List[str] = ["time"],
        normalizer: Optional[Normalizer] = None,
    ) -> xr.Dataset:
        if self.mode == "train":
            self.normalizer = Normalizer(fit_ds=ds, collapse_dims=collapse_dims)
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
(
        self,
        pixel: str,
        x_d: np.ndarray,
        y: np.ndarray,
        times: Optional[np.ndarray] = None,
        x_s: Optional[np.ndarray] = None,
    ):
        self.x_d[pixel] = x_d
        self.y[pixel] = y
        if self.static_inputs is not None:
            self.x_s[pixel] = x_s

        # store metadata
        if times is not None:
            #  NOTE: this is super inefficient becuase duplicated over each pixel
            #  store as integers to keep pytorch happy
            self.times[pixel] = np.array(times).astype(np.int64)

    def create_lookup_table(self, ds):
        pixels_without_samples = []
        lookup: List[Tuple[str, int]] = []

        for pixel in tqdm(ds.sample.values, desc="Loading Data: "):
            df_native = ds.sel(sample=pixel).to_dataframe()
            self.freq = _check_no_missing_times_in_time_series(df_native)

            #  store times as float
            times = df_native.index

            # TODO: Include forecasted variables into dynamic
            # self.forecast_variables

            #  Shift values one timestep to ensure no leakage
            # TODO: do we need to do this? or do we do in __getitem__
            # if self.horizon > 0:
            #     x_d = df_native[self.inputs].shift(self.horizon).values
            # else:
            #     x_d = df_native[self.inputs].values

            x_d = df_native[self.inputs].values
            y = df_native[self.target].values

            #  TODO: deal with static inputs
            if self.static_inputs is not None:
                # index = pixel; columns = variables
                x_s = df_static.loc[pixel, self.static_inputs].values
            else:
                x_s = None

            # checks inputs and outputs for each sequence. valid: flag = 1, invalid: flag = 0
            flag = validate_samples(
                x_d=x_d,
                x_s=x_s,
                y=y,
                seq_length=self.seq_length,
                forecast_horizon=self.horizon,
            )

            # store lookups with Pixel_ID and Index for that valid sample
            valid_samples = np.argwhere(flag == 1)
            [lookup.append((pixel, smp)) for smp in valid_samples]

            # store data if basin has at least ONE valid sample
            if valid_samples.size > 0:(pixel, x_d=x_d, y=y, x_s=x_s, times=times)
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
        pixel, target_index = self.lookup_table[idx]
        target_index = int(target_index)
        data = {}

        #  Get input/output data
        #  get the inputs for [target - seq_length : target - horizon]
        x_d = (
            torch.from_numpy(
                self.x_d[pixel][
                    ((target_index - self.seq_length - self.horizon) + 1) : (
                        target_index - self.horizon
                    )
                    + 1
                ]
            )
            .float()
            .to(self.device)
        )
        if self.DEBUG:
            assert x_d.shape[0] == self.cfg.seq_length

        # get target for current : horizon
        # if self.horizon == 0:
        #     #  simulate the current timestep
        #     y_ = self.y[pixel][target_index].reshape(-1, 1)
        #     time_ = self.times[pixel][target_index]
        # else:
        #  forecast the next `self.horizon` timesteps
        end_fcast_correction = 1 if self.horizon == 0 else 0
        y_ = self.y[pixel][
            target_index : (target_index + self.horizon + end_fcast_correction)
        ]
        y_ = y_.reshape(-1, 1) if y_.ndim == 1 else y_
        time_ = self.times[pixel][target_index : (target_index + self.horizon)]

        y = torch.from_numpy(y_).float().to(self.device)

        if self.static_inputs is not None:
            x_s = torch.cat(self.x_s[pixel], dim=-1).float().to(self.device)
        else:
            x_s = torch.from_numpy(np.array([])).float().to(self.device)

        # metadata, store time as integer64
        # convert back to timestamp https://stackoverflow.com/a/47562725/9940782
        time_ = np.array(time_) if not isinstance(time_, np.ndarray) else time_
        time = torch.from_numpy(time_).float().to(self.device)
        target_index = (
            torch.from_numpy(np.array([idx]).reshape(-1)).float().to(self.device)
        )

        # write output dictionary
        meta = {
            "index": target_index,
            "target_time": time,
        }

        data["x_d"] = x_d
        data["y"] = y
        data["meta"] = meta
        data["x_s"] = x_s

        return data
