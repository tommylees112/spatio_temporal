import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import pytest
import xarray as xr

#  import from pipeline
from tests.utils import (
    _make_dataset,
    get_oxford_weather_data,
    load_test_jena_data_as_dataset,
    create_and_assign_temp_run_path_to_config,
    create_sin_with_different_phases,
)
from spatio_temporal.data.dataloader import (
    XarrayDataset,
    PixelDataLoader,
)
from spatio_temporal.data.data_utils import _stack_xarray
from spatio_temporal.config import Config

TEST_REAL_DATA = False

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
        # check that works on 1D data too ...

    def test_dataset(self, tmp_path):
        target_variable = "target"
        input_variables = ["feature"]
        pixel_dims = ["lat", "lon"]
        cfg = Config(Path("tests/testconfigs/test_config.yml"))

        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
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

    def test_dataloader(self, tmp_path):
        ds = _make_dataset()
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        dl = PixelDataLoader(
            ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size
        )

        assert dl.batch_size == cfg.batch_size

        batch_size = 30
        seq_length = 10
        autoregressive = cfg.autoregressive
        data = next(iter(dl))
        assert len(data) == 2, "Expected X, y samples, len == 2"
        n_inputs = len(["features"]) + 1 if autoregressive else len(["features"])

        assert data[0].shape == (
            batch_size,
            seq_length,
            n_inputs,
        ), f"Size Mismatch! Expected: {(batch_size, seq_length, n_inputs)} Got: {data[0].shape}"

    def test_1D_data(self, tmp_path):
        # convert pandas to xarray object
        ds = load_test_jena_data_as_dataset()
        cfg = Config(Path("tests/testconfigs/test_1d_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)

        dl = PixelDataLoader(
            ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size
        )

        x, y = dl.__iter__().__next__()

        assert x.shape == (cfg.batch_size, cfg.seq_length, len(cfg.input_variables))
        assert y.shape == (cfg.batch_size, cfg.seq_length, 1)

    def test_kenya_data(self, tmp_path):
        if TEST_REAL_DATA:
            ds = pickle.load(Path("data/kenya.pkl").open("rb"))
            cfg = Config(Path("tests/testconfigs/config.yml"))
            create_and_assign_temp_run_path_to_config(cfg, tmp_path)

            dl = PixelDataLoader(
                ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size
            )

            data = dl.__iter__().__next__()

            batch_size = 256
            seq_length = 10
            input_variables = ["precip", "t2m", "SMsurf"]
            autoregressive = True
            n_inputs = len(input_variables) + 1 if autoregressive else len(input_variables)

            assert cfg.batch_size == batch_size
            assert cfg.autoregressive == autoregressive
            assert data[0].shape == (
                batch_size,
                seq_length,
                n_inputs,
            ), f"X Data Mismatch! Expected: {(batch_size, seq_length, n_inputs)} Got: {data[0].shape}"
        else:
            pass

    def test_runoff_data(self, tmp_path):
        if TEST_REAL_DATA:
            ds = xr.open_dataset("data/ALL_dynamic_ds.nc")
            cfg = Config(Path("tests/testconfigs/config_runoff.yml"))
            create_and_assign_temp_run_path_to_config(cfg, tmp_path)
            
            # train period
            train_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
                time=slice(cfg.train_start_date, cfg.train_end_date)
            )
            train_dl = PixelDataLoader(train_ds, cfg=cfg, mode="train", num_workers=4, batch_size=cfg.batch_size)
            
            # check data is loaded properly
            x, y = next(iter(train_dl))
            n_in_vars = len(cfg.input_variables) + 1 if cfg.autoregressive else len(cfg.input_variables)
            assert x.shape == (cfg.batch_size, cfg.seq_length, n_in_vars)
            assert y.shape == (cfg.batch_size, cfg.seq_length, 1)
        else:
            pass


    def test_sine_wave_example():
        #  create_sin_with_different_phases()
        pass

    def test_longer_horizon_fcast(self, tmp_path):
        cfg = Config(Path("tests/testconfigs/test_1d_config_horizon.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        ds = load_test_jena_data_as_dataset()

        dl = PixelDataLoader(
            ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size
        )
        x, y = dl.__iter__().__next__()
        assert y.shape == (cfg.batch_size, cfg.seq_length, cfg.horizon)
        assert False

    def test_static_inputs(self, tmp_path):
        ds = _make_dataset()
        ds_static = ds.mean(dim="time")

        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        assert False

    def test_forecast_inputs(self, tmp_path):
        ds = _make_dataset()
        ds_forecast = ds.shift(1).drop("target").rename({"feature": "feature_fcast1"})
        ds = xr.merge([ds, ds_forecast])

        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        assert False
