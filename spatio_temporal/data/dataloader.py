""" https://discuss.pytorch.org/t/correctly-feeding-lstm-with-minibatch-time-sequence-data/52101/26
    https://discuss.pytorch.org/t/deadlock-with-dataloader-and-xarray-dask/9387
    https://phausamann.github.io/sklearn-xarray/content/target.html 
    https://phausamann.github.io/sklearn-xarray/auto_examples/plot_activity_recognition.html#sphx-glr-auto-examples-plot-activity-recognition-py
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
import xarray as xr
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from spatio_temporal.data.normaliser import Normalizer
from spatio_temporal.config import Config
from spatio_temporal.data.dataset import XarrayDataset


class PixelDataLoader(DataLoader):
    """A simple torch DataLoader wrapping xr.DataArray"""

    def __init__(
        self,
        data: xr.Dataset,
        cfg: Config,
        mode: str,
        normalizer: Optional[Normalizer] = None,
        DEBUG: bool = False,
        **kwargs,
    ):
        #  TODO: add ability to create different subsets by time (train, test, validation)
        dataset = XarrayDataset(data, cfg=cfg, mode=mode, normalizer=normalizer, DEBUG=DEBUG)
        super().__init__(dataset, **kwargs)

        self.input_size = dataset.input_size
        self.output_size = dataset.output_size
        self.mode = mode
        self.normalizer = dataset.normalizer
        self.freq = dataset.freq
        self.horizon = dataset.horizon
