from pathlib import Path

from tests.utils import (
    create_linear_ds,
    create_and_assign_temp_run_path_to_config,
    _create_dummy_normalizer,
    get_pollution_data_beijing,
)
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.training.tester import Tester
from spatio_temporal.config import Config


class TestTrainer:
    def test_trainer(self, tmp_path: Path):
        ds = create_linear_ds().isel(lat=slice(0, 5), lon=slice(0, 5))
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        cfg.run_dir = tmp_path
        Trainer(cfg=cfg, ds=ds)

        #  TODO: test the trainer training loop
        #  TODO: test that plots created, outputs saved
        #  TODO: test that the correct period data are loaded into dataloaders
        #  TODO: test that the correct data comes through iteration over datalaoder

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

        assert False


class TestTester:
    def test_tester(self, tmp_path):
        ds = create_linear_ds().isel(lat=slice(0, 5), lon=slice(0, 5))
        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)

        # initialise the train directory!
        # expecting a `config.yml` & `normalizer.pkl`
        cfg.dump_config(cfg.run_dir)
        _create_dummy_normalizer(ds=ds, cfg=cfg)

        Tester(cfg=cfg, ds=ds)

        #  TODO: test the tester evaluation loop
        #  TODO: test that plots created, outputs saved
        #  TODO: test that the correct period data are loaded into dataloaders
        #  TODO: test that the correct data comes through iteration over datalaoder
