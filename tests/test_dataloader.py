import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import pytest

#  import from pipeline
from tests.utils import _make_dataset, get_oxford_weather_data
from spatio_temporal.data.dataloader import (
    XarrayDataset,
    PixelDataLoader,
)
from spatio_temporal.data.data_utils import _stack_xarray
from spatio_temporal.config import Config


class TestDataLoader:
    def test_stack_xarray(self):
        ds = _make_dataset()
        stacked, sample = _stack_xarray(ds, spatial_coords=["lat", "lon"])

        #  check that stacking works
        unstacked = sample.unstack()
        pixel = unstacked.isel(
            lat=np.random.choice(len(unstacked["lat"].values)),
            lon=np.random.choice(len(unstacked["lon"].values)),
        )
        lat, lon = [float(ll) for ll in str(pixel.values).split("_")]

        assert np.allclose(
            [lat, lon], [float(pixel.lat.values), float(pixel.lon.values)]
        )

        # check that can recreate original dataset

    def test_dataset(self):
        target_variable = "target"
        input_variables = ["feature"]
        pixel_dims = ["lat", "lon"]
        cfg = Config(Path("tests/testconfigs/test_config.yml"))

        raw_ds = _make_dataset()
        ds = XarrayDataset(raw_ds, cfg=cfg, mode="train")

        assert ds.target == target_variable
        assert (
            ds.inputs == input_variables + ["autoregressive"]
            if cfg.autoregressive
            else input_variables
        )

        x_features = (
            len(input_variables) + 1 if cfg.autoregressive else len(input_variables)
        )
        seq_length = 10
        for i in range(10):
            x, y = ds.__getitem__(i)

            assert y.shape == (seq_length, 1)
            assert x.shape == (
                seq_length,
                x_features,
            ), f"Shape Mismatch! Expect: {(seq_length, x_features)} Got: {x.shape}"

    def test_dataloader(self):
        ds = _make_dataset()
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        dl = PixelDataLoader(
            ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size
        )

        assert dl.batch_size == cfg.batch_size

        batch_size = 30
        seq_length = 10
        autoregressive = False
        data = next(iter(dl))
        assert len(data) == 2, "Expected X, y samples, len == 2"
        n_inputs = len(["features"]) + 1 if autoregressive else len(["features"])

        assert data[0].shape == (
            batch_size,
            seq_length,
            n_inputs,
        ), f"Size Mismatch! Expected: {(batch_size, seq_length, n_inputs)} Got: {data[0].shape}"

    def test_kenya_data(self):
        ds = pickle.load(Path("data/kenya.pkl").open("rb"))
        cfg = Config(Path("tests/testconfigs/config.yml"))

        dl = PixelDataLoader(
            ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size
        )

        data = dl.__iter__().__next__()

        batch_size = 256
        seq_length = 10
        input_variables = ["precip", "t2m", "SMsurf"]
        autoregressive = False
        n_inputs = len(input_variables) + 1 if autoregressive else len(input_variables)

        assert cfg.batch_size == batch_size
        assert cfg.autoregressive == autoregressive
        assert data[0].shape == (
            batch_size,
            seq_length,
            n_inputs,
        ), f"X Data Mismatch! Expected: {(batch_size, seq_length, n_inputs)} Got: {data[0].shape}"
