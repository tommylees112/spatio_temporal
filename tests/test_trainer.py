from pathlib import Path
import pandas as pd
import xarray as xr
from pandas.tseries.offsets import DateOffset, MonthEnd
from tests.utils import (
    create_linear_ds,
    create_and_assign_temp_run_path_to_config,
    _create_dummy_normalizer,
    get_pollution_data_beijing,
)
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.training.tester import Tester
from spatio_temporal.config import Config
from spatio_temporal.data.data_utils import train_test_split


class TestTrainer:
    def test_trainer(self, tmp_path: Path):
        ds = create_linear_ds().isel(lat=slice(0, 5), lon=slice(0, 5))
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        cfg.run_dir = tmp_path
        Trainer(cfg=cfg, ds=ds)

        #  TODO: test the trainer training loop:: trainer._train_one_epoch
        #  TODO: test that plots created, outputs saved:: model.pt / optimizer.pt
        #  TODO: test that the correct period data are loaded into dataloaders::
        #  TODO: test that the correct data comes through iteration over datalaoder

    def test_train_test_split(self, tmp_path):
        ds = create_linear_ds().isel(lat=slice(0, 5), lon=slice(0, 5))
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        cfg.run_dir = tmp_path

        train = train_test_split(ds, cfg, subset="train")
        test = train_test_split(ds, cfg, subset="test")
        valid = train_test_split(ds, cfg, subset="validation")

        cfg.train_start_date
        cfg.train_end_date
        cfg.validation_start_date
        cfg.validation_end_date
        cfg.test_start_date
        cfg.test_end_date

    def test_pollution(self, tmp_path):
        ds = get_pollution_data_beijing().to_xarray()
        cfg = Config(cfg_path=Path("tests/testconfigs/pollution.yml"))
        cfg.run_dir = tmp_path
        trainer = Trainer(cfg, ds)

        train_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
            time=slice(cfg.train_start_date, cfg.train_end_date)
        )

        assert trainer.train_dl.dataset.lookup_table != {}
        assert trainer.train_dl.dataset.y != {}
        assert trainer.train_dl.dataset.x_d != {}


class TestTester:
    def test_tester(self, tmp_path):
        ds = create_linear_ds().isel(lat=slice(0, 5), lon=slice(0, 5))
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        cfg._cfg["n_epochs"] = 1
        cfg._cfg["num_workers"] = 1
        cfg._cfg["horizon"] = 5
        cfg.run_dir = tmp_path

        # initialise the train directory!
        trainer = Trainer(cfg, ds)
        trainer.train_and_validate()

        tester = Tester(cfg=cfg, ds=ds)

        #  TODO: test the tester evaluation loop
        tester.run_test()
        #  TODO: test that plots created, outputs saved
        outfile = sorted(list(cfg.run_dir.glob("*.nc")))[-1]
        out_ds = xr.open_dataset(outfile)

        assert len(out_ds.horizon.values) == cfg.horizon

        #  Check that the times are correct
        min_time = pd.to_datetime(out_ds.time.values.min())
        exp_min_time = cfg.test_start_date + DateOffset(
            months=(cfg.seq_length + cfg.horizon) - 1
        )

        assert all(
            [
                (min_time.year == exp_min_time.year),
                (min_time.month == exp_min_time.month),
                (min_time.day == exp_min_time.day),
            ]
        )

        max_time = pd.to_datetime(out_ds.time.values.max()) + MonthEnd(-1)
        exp_max_time = cfg.test_end_date - DateOffset(months=1)

        assert all(
            [
                (max_time.year == exp_max_time.year),
                (max_time.month == exp_max_time.month),
                (max_time.day == exp_max_time.day),
            ]
        )

        #  TODO: test that the correct period data are loaded into dataloaders
        #  TODO: test that the correct data comes through iteration over datalaoder
