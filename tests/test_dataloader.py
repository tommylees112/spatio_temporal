import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import pytest
import xarray as xr
from pandas.tseries.offsets import DateOffset

#  import from pipeline
from tests.utils import (
    _make_dataset,
    get_oxford_weather_data,
    load_test_jena_data_as_dataset,
    create_and_assign_temp_run_path_to_config,
    create_sin_with_different_phases,
    create_linear_ds,
    get_pollution_data_beijing,
)
from spatio_temporal.data.data_utils import (
    load_all_data_from_dl_into_memory,
    validate_samples,
)
from spatio_temporal.data.dataloader import (
    XarrayDataset,
    PixelDataLoader,
)
from spatio_temporal.data.data_utils import _stack_xarray
from spatio_temporal.config import Config

TEST_REAL_DATA = True


class TestDataLoader:
    # TODO: TEST FOR scrambled labels / mixing the features and timesteps axes / things like this
    def test_stack_xarray(self):
        ds = _make_dataset().isel(lat=slice(0, 2), lon=slice(0, 1))
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

    def test_correct_data_returned(self, tmp_path):
        ds = _make_dataset().isel(lat=slice(0, 2), lon=slice(0, 1))
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        cfg._cfg["encode_doys"] = True
        cfg._cfg["static_inputs"] = "embedding"
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        dl = PixelDataLoader(ds, cfg=cfg, mode="train", DEBUG=True)
        data = dl.__iter__().__next__()
        x, y = data["x_d"], data["y"]

        #  recreate the stacked dataset
        stacked_ds = dl.dataset.ds

        #  get the current_time_index and pixel from the __getitem__() call
        getitem_call = int(data["meta"]["index"])
        pixel, current_time_index = dl.dataset.lookup_table[getitem_call]

        # check that the returned data is valid
        #  TODO: wrap into function for getting the valid times!
        est_target_time = pd.to_datetime(
            np.array(data["meta"]["target_time"]).astype("datetime64[ns]")
        )[0]

        #  rounding error because of storing as float
        input_data_times = pd.to_datetime(stacked_ds.time.values)
        true_target_index = input_data_times.get_loc(est_target_time, method="nearest")
        true_target_time = input_data_times[true_target_index]

        assert current_time_index + cfg.horizon == true_target_index

        # :: RECREATE TARGET DATA ::
        all_expected_y = stacked_ds.sel(sample=pixel)["target"].values

        expected_y = stacked_ds.sel(sample=pixel, time=true_target_time)[
            cfg.target_variable
        ].values
        expected_y_index = (
            stacked_ds.sel(sample=pixel)
            .isel(time=true_target_index)[cfg.target_variable]
            .values
        )
        assert expected_y == expected_y_index
        assert np.isclose(y.flatten()[-1], expected_y)

        ## :: RECREATE INPUT DATA ::
        # max_input_ix should be the CURRENT TIME (+ 1 because of exlusive upper indexing)
        max_input_ix = int(true_target_index - cfg.horizon)
        assert max_input_ix == current_time_index
        max_input_time = input_data_times[max_input_ix]

        #  min_input_ix = the first input time
        min_input_ix = int(max_input_ix - cfg.seq_length) + 1
        min_input_time = input_data_times[min_input_ix]

        input_vars = (
            cfg.input_variables + ["autoregressive"]
            if cfg.autoregressive
            else cfg.input_variables
        )

        # has x been drawn from the actual underlying data?
        all_expected_x = stacked_ds.sel(sample=pixel)["feature"].values
        _expected_x = all_expected_x[min_input_ix:max_input_ix]
        # assert x == _expected_x

        # assert all(
        #     np.isin(
        #         np.round(x.numpy().flatten(), 3).astype("float64"),
        #         np.round(all_expected_x.flatten(), 3).astype("float64"),
        #     )
        # )

        # get the exact expected input vector
        # NOTE: slice is NOT EXCLUSIVE UPPER therefore need to exclude the final
        expected_x_feature = (
            stacked_ds.sel(sample=pixel, time=slice(min_input_time, max_input_time))[
                input_vars
            ]
            .to_array()
            .values.T
        )

        x_feature = np.array(x)
        x_feature = x_feature.reshape(expected_x_feature.shape)

        assert np.allclose(x_feature, expected_x_feature)

    def test_dataset_beijing(self, tmp_path):
        path = Path("tests/testconfigs/pollution.yml")
        cfg = Config(path)
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        raw_ds = get_pollution_data_beijing().to_xarray().isel(time=slice(0, 1000))
        ds = XarrayDataset(raw_ds, cfg=cfg, mode="train", DEBUG=True)

        assert ds.y != {}

    def test_dataset(self, tmp_path):
        target_variable = "target"
        input_variables = ["feature"]
        for path in [
            Path("tests/testconfigs/test_config_simulate.yml"),
            Path("tests/testconfigs/test_config.yml"),
        ]:
            cfg = Config(path)

            create_and_assign_temp_run_path_to_config(cfg, tmp_path)
            raw_ds = _make_dataset().isel(lat=slice(0, 2), lon=slice(0, 1))
            ds = XarrayDataset(raw_ds, cfg=cfg, mode="train", DEBUG=True)

            assert ds.target == target_variable
            assert (
                ds.inputs == input_variables + ["autoregressive"]
                if cfg.autoregressive
                else input_variables
            )

            x_features = (
                len(input_variables) + 1 if cfg.autoregressive else len(input_variables)
            )
            seq_length = cfg.seq_length
            for i in range(10):
                data = ds.__getitem__(i)
                x, y = data["x_d"], data["y"]

                assert y.shape == (1, 1)
                assert x.shape == (
                    seq_length,
                    x_features,
                ), f"Shape Mismatch! Expect: {(seq_length, x_features)} Got: {x.shape}"

                meta = data["meta"]
                times = (
                    meta["target_time"]
                    .detach()
                    .numpy()
                    .astype("datetime64[ns]")
                    .flatten()
                )
                pixel, _ = ds.lookup_table[int(meta["index"])]
                latlon = tuple([float(l) for l in str(pixel).split("_")])

                y_unnorm = (
                    ds.normalizer.individual_inverse(y, pixel, variable="target")
                    .detach()
                    .numpy()
                )

                #  extract from the original xr.Dataset
                y_exp = raw_ds.sel(
                    lat=latlon[0], lon=latlon[1], time=times, method="nearest"
                )[cfg.target_variable].values
                assert np.isclose(y_unnorm.reshape(y_exp.shape), y_exp, atol=1e-5)

    def test_dataloader(self, tmp_path):
        ds = _make_dataset()
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        dl = PixelDataLoader(
            ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size,
        )

        assert dl.batch_size == cfg.batch_size

        batch_size = 30
        seq_length = cfg.seq_length
        autoregressive = cfg.autoregressive
        data = next(iter(dl))
        x, y = data["x_d"], data["y"]
        n_inputs = len(["features"]) + 1 if autoregressive else len(["features"])

        assert x.shape == (
            batch_size,
            seq_length,
            n_inputs,
        ), f"Size Mismatch! Expected: {(batch_size, seq_length, n_inputs)} Got: {x.shape}"

    def test_1D_data(self, tmp_path):
        # convert pandas to xarray object
        ds = load_test_jena_data_as_dataset()
        cfg = Config(Path("tests/testconfigs/test_1d_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)

        dl = PixelDataLoader(
            ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size
        )

        data = dl.__iter__().__next__()
        x, y = data["x_d"], data["y"]

        assert x.shape == (cfg.batch_size, cfg.seq_length, len(cfg.input_variables))
        assert y.shape == (cfg.batch_size, 1, 1)

    def test_kenya_data(self, tmp_path):
        if TEST_REAL_DATA:
            ds = pickle.load(Path("data/kenya.pkl").open("rb")).isel(
                lat=slice(0, 5), lon=slice(0, 5)
            )
            cfg = Config(Path("tests/testconfigs/config.yml"))
            create_and_assign_temp_run_path_to_config(cfg, tmp_path)

            dl = PixelDataLoader(
                ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size
            )

            data = dl.__iter__().__next__()
            x, _ = data["x_d"], data["y"]

            batch_size = 256
            seq_length = cfg.seq_length
            input_variables = ["precip", "t2m", "SMsurf"]
            autoregressive = True
            n_inputs = (
                len(input_variables) + 1 if autoregressive else len(input_variables)
            )

            assert cfg.batch_size == batch_size
            assert cfg.autoregressive == autoregressive
            assert x.shape == (
                batch_size,
                seq_length,
                n_inputs,
            ), f"X Data Mismatch! Expected: {(batch_size, seq_length, n_inputs)} Got: {x.shape}"
        else:
            pass

    def test_runoff_data(self, tmp_path):
        if TEST_REAL_DATA:
            ds = xr.open_dataset("data/ALL_dynamic_ds.nc").isel(station_id=slice(0, 5))
            cfg = Config(Path("tests/testconfigs/config_runoff.yml"))
            create_and_assign_temp_run_path_to_config(cfg, tmp_path)

            # train period
            train_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
                time=slice(cfg.train_start_date, cfg.train_end_date)
            )
            train_dl = PixelDataLoader(
                train_ds,
                cfg=cfg,
                mode="train",
                num_workers=4,
                batch_size=cfg.batch_size,
            )

            #  check data is loaded properly
            data = next(iter(train_dl))
            x, y = data["x_d"], data["y"]

            n_in_vars = (
                len(cfg.input_variables) + 1
                if cfg.autoregressive
                else len(cfg.input_variables)
            )
            assert x.shape == (cfg.batch_size, cfg.seq_length, n_in_vars)
            assert y.shape == (cfg.batch_size, 1, 1)
        else:
            pass

    def test_linear_example(self, tmp_path):
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)

        alpha = 0
        beta = 2
        epsilon_sigma = 0

        ds = create_linear_ds(
            horizon=cfg.horizon, alpha=alpha, beta=beta, epsilon_sigma=epsilon_sigma
        ).isel(lat=slice(0, 2), lon=slice(0, 2))
        dl = PixelDataLoader(
            ds,
            cfg=cfg,
            num_workers=1,
            mode="train",
            batch_size=cfg.batch_size,
            DEBUG=True,
        )

        #  load all of the data into memory
        data = load_all_data_from_dl_into_memory(dl)
        x = data["x_d"]
        assert x.shape[-1] == cfg.seq_length
        y = data["y"]
        times = pd.to_datetime(data["time"].astype("datetime64[ns]").flatten())

        # matching batch dims for all samples
        assert x.shape[0] == y.shape[0]

        #  test ONE SINGLE (x, y) sample
        INDEX = 1

        # get metadata for sample
        idx = int(data["index"][INDEX])
        pixel, valid_current_time_index = dl.dataset.lookup_table[idx]
        latlon = tuple([float(l) for l in str(pixel).split("_")])
        target_time = times[INDEX]
        current_time = times[valid_current_time_index][0]

        #  get the correct times (weird indexing becuase of imperfect translation of float -> datetime64[ns])
        max_time = target_time - DateOffset(months=cfg.horizon) + DateOffset(days=2)
        min_time = max_time - DateOffset(months=cfg.seq_length)
        input_times = pd.date_range(min_time, max_time, freq="M")[-cfg.seq_length :]

        #  recreate the data that should be loaded from the raw xr.Dataset
        stacked, _ = _stack_xarray(ds, spatial_coords=cfg.pixel_dims)
        normalizer = dl.normalizer
        norm_stacked = normalizer.transform(stacked)

        all_y = norm_stacked["target"].sel(sample=pixel)
        _y = all_y.sel(time=target_time, method="nearest")
        all_x = norm_stacked["feature"].sel(sample=pixel)
        _x_d = all_x.sel(time=input_times, method="nearest")

        #  check that the dataloader saves & returns the correct values
        assert np.allclose(
            dl.dataset.y[pixel], (all_y.values)
        ), "The DataLoader saves incorrect y values to memory"
        assert np.isclose(
            _y.values, y[INDEX]
        ), "The DataLoader returns an incorrect value from the Dataset"

        #  input (X) data
        dataset_loaded = dl.dataset.x_d[pixel]
        expected = all_x.values.reshape(dataset_loaded.shape)
        mask = np.isnan(expected)
        expected = expected[~mask]
        dataset_loaded = dataset_loaded[~mask]

        assert np.allclose(
            dataset_loaded, expected
        ), f"The dataloader is saving the wrong data to the lookup table. {dataset_loaded[:10]} {expected[:10]}"

        #  get input X data from INDEX (not times)
        max_input_ix = int(valid_current_time_index)
        min_input_ix = int(max_input_ix - cfg.seq_length) + 1
        _x_d_index_values = all_x.values[min_input_ix : max_input_ix + 1]

        assert np.allclose(_x_d_index_values, _x_d.values)

        # TODO: Why does this not work?
        if False:
            assert np.allclose(
                _x_d_index_values.values, x[INDEX]
            ), "The Dynamic Data is not the data we expect"

        #  check that the raw data is the linear combination we expect
        # "target" should be linear combination of previous timestep "feature"
        # (y = x @ [0, 2])
        zeros = np.zeros((cfg.seq_length - 1, 1))
        betas = np.append(zeros, beta).reshape(-1, 1)
        unnorm_x = dl.dataset.normalizer.individual_inverse(
            x[INDEX], pixel_id=pixel, variable=cfg.input_variables[0]
        )
        unnorm_y = dl.dataset.normalizer.individual_inverse(
            y[INDEX], pixel_id=pixel, variable=cfg.target_variable
        )

        #  time=target_time,
        ds.sel(lat=latlon[0], lon=latlon[1], method="nearest")[cfg.target_variable]
        assert unnorm_x @ betas == unnorm_y

        #  TODO: what would be the error in the normalized space
        # y_hat = x @ betas
        # assert np.allclose(y_hat, y)

    def test_sine_wave_example(self):
        #  create_sin_with_different_phases()
        pass

    def test_longer_horizon_fcast(self, tmp_path):
        cfg = Config(Path("tests/testconfigs/test_1d_config_horizon.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        ds = load_test_jena_data_as_dataset()

        dl = PixelDataLoader(
            ds, cfg=cfg, num_workers=1, mode="train", batch_size=cfg.batch_size
        )
        data = dl.__iter__().__next__()
        _, y = data["x_d"], data["y"]

        assert y.shape == (cfg.batch_size, 1, 1)

    def test_static_inputs(self, tmp_path):
        ds = _make_dataset().isel(lat=slice(0, 2), lon=slice(0, 1))
        ds_static = ds.mean(dim="time")

        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        assert False

    def test_forecast_inputs(self, tmp_path):
        ds = _make_dataset().isel(lat=slice(0, 2), lon=slice(0, 1))
        ds_forecast = (
            ds.shift(time=1).rename({"feature": "feature_fcast1"}).drop("target")
        )
        ds = xr.merge([ds, ds_forecast])

        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        assert False

    def test_validate_samples(self):
        #  create data with nans
        #  ensure that the validate_samples correctly ignores them based on these criteria
        #  1. not enough history (seq_length > history)
        #  5. not enough data for forecast horizon
        #  2. NaN in the dynamic inputs
        #  3. NaN in the outputs (only for training period)
        #  4. any NaN in the static features makes the target_index invalid
        validate_samples
        assert False
