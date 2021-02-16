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

    def __init__(self, data: xr.Dataset, cfg: Config, mode: str):
        self.cfg = cfg
        #  TODO: how to keep track of metadata
        # stack pixel_dims to one dimension ("sample")
        stacked, sample = _stack_xarray(data, self.cfg.pixel_dims)

        self.mode = mode
        self.sample: xr.Dataset = sample
        self.batch_dim: str = "sample"
        self.seq_length: int = self.cfg.seq_length
        self.inputs: List[str] = self.cfg.input_variables
        self.target: str = self.cfg.target_variable

        # TODO: allow static inputs
        self.static_inputs = cfg.static_inputs

        # TODO: allow forecast variables
        self.forecast_variables = self.cfg.forecast_variables

        ds: xr.Dataset = stacked
        ds = self.run_normalisation(ds)

        # add autoregressive variable
        if cfg.autoregressive:
            auto = stacked[self.target].shift(time=1).rename("autoregressive")
            self.inputs = self.inputs + ["autoregressive"]
            ds = xr.merge([ds, auto])

        self.input_size = len(self.inputs)
        #  TODO: work with multiple time horizons (1, ..., h)
        self.horizon = cfg.horizon
        self.output_size = 1  # self.horizon

        # init dictionaries to store the RAW pixel timeseries
        self.x_d: Dict[str, Tuple[str, int]] = {}
        self.y: Dict[str, Tuple[str, int]] = {}
        self.x_s: Dict[str, Optional[Tuple[str, int]]] = {}

        # 1. Check for missing data
        # 2. Store int -> data index
        # 3. Store y, x_d, x_s
        self.create_lookup_table(ds)

    def run_normalisation(self, ds, collapse_dims: List[str] = ["time"]) -> xr.Dataset:
        if self.mode == "train":
            self.normalizer = Normalizer(fit_ds=ds, collapse_dims=collapse_dims)
            pickle.dump(
                self.normalizer, (self.cfg.run_dir / "normalizer.pkl").open("wb")
            )
        else:
            self.normalizer = pickle.load(
                (self.cfg.run_dir / "normalizer.pkl").open("rb")
            )
            ds = self.normalizer.transform(ds)
        return ds

    def store_data(
        self,
        pixel: str,
        x_d: np.ndarray,
        y: np.ndarray,
        x_s: Optional[np.ndarray] = None,
    ):
        self.x_d[pixel] = x_d
        self.y[pixel] = y
        if self.static_inputs is not None:
            self.x_s[pixel] = x_s

    def create_lookup_table(self, ds):
        # self.freq = pd.infer_freq(self.ds.time.values)
        pixels_without_samples = []
        lookup: List[Tuple[str, int]] = []

        for pixel in tqdm(ds.sample.values, desc="Loading Data: "):
            df_native = ds.sel(sample=pixel).to_dataframe()
            _check_no_missing_times_in_time_series(df_native)

            # TODO: Include forecasted variables into dynamic
            # self.forecast_variables
            x_d = df_native[self.inputs].values

            # TODO: forecast horizons != 1
            # self.horizon
            y = df_native[self.target].values

            #  TODO: deal with static inputs
            if self.static_inputs is not None:
                # index = pixel; columns = variables
                x_s = df_static.loc[pixel, self.static_inputs].values
            else:
                x_s = None

            # checks inputs and outputs for each sequence. valid: flag = 1, invalid: flag = 0
            flag = validate_samples(x_d, x_s, y, seq_length=self.seq_length)

            # store lookups with Pixel_ID and Index for that valid sample
            valid_samples = np.argwhere(flag == 1)
            [lookup.append((pixel, smp)) for smp in valid_samples]

            # store data if basin has at least ONE valid sample
            if valid_samples.size > 0:
                self.store_data(pixel, x_d=x_d, y=y, x_s=x_s)
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
        pixel, index = self.lookup_table[idx]
        index = int(index)

        data = {}
        data["x_d"] = torch.from_numpy(
            self.x_d[pixel][index - self.seq_length + 1 : index + 1]
        ).float()
        data["y"] = torch.from_numpy(
            self.y[pixel][index - self.seq_length + 1 : index + 1].reshape(-1, 1)
        ).float()

        if self.static_inputs is not None:
            data["x_s"] = torch.cat(self.x_s[pixel], dim=-1).float()

        return data["x_d"], data["y"]
