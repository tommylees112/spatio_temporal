from pathlib import Path
from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.training.tester import Tester
from tests.utils import create_linear_ds, create_static_example_data
from spatio_temporal.training.eval_utils import save_loss_curves, save_timeseries
from run import load_data
from tests.utils import create_and_assign_temp_run_path_to_config
from typing import Union
import numpy as np
import xarray as xr
from pprint import pformat
from datetime import datetime


def create_test_dir() -> Path:
    now = datetime.now()
    dir_name = f"pytest-{now.month}{now.day}{now.minute}{now.second}{np.random.randint(1, 100)}"
    try:
        pytest_dir = Path(
            "/private/var/folders/q3/0lmt64ld10s14_0n0vxpt_m00000gp/T/pytest-of-tommylees"
        )
        tmp_path = pytest_dir / dir_name
        tmp_path.mkdir(exist_ok=True, parents=True)
    except PermissionError as E:
        pytest_dir = Path("runs/TESTS/")
        tmp_path = pytest_dir / dir_name
        tmp_path.mkdir(exist_ok=True, parents=True)
    return tmp_path


class TestPipeline:
    @staticmethod
    def check_loaded_data(
        cfg: Config,
        trainer_tester: Union[Trainer, Tester],
        data: xr.Dataset,
        is_train: bool = True,
    ):
        if is_train:
            dataloader = trainer_tester.train_dl
        else:
            dataloader = trainer_tester.test_dl

        pixels = [k for k in dataloader.dataset.x_s.keys()]
        pixel = np.random.choice(pixels)

        if cfg.static_inputs is not None:
            #  check static size
            assert dataloader.dataset.x_s[pixel].numpy().shape == (
                len(cfg.static_inputs),
            )

        # check dynamic size
        assert dataloader.dataset.x_d[pixel].numpy().shape == (
            int(data.time.values.shape[0]),
            len(cfg.input_variables),
        )

    @staticmethod
    def check_output_files(tmp_path: Path):
        #  check the saved files (model and optimizer epochs)
        test_dir = sorted([d for d in tmp_path.glob("runs/test*")])[-1]
        created_files = sorted([t.name for t in test_dir.iterdir()])

        cfg = Config(test_dir / "config.yml")
        assert len([f for f in created_files if "model_epoch" in f]) == cfg.n_epochs

        # check the normalizers created
        if (cfg.static_inputs is not None) and (cfg.static_normalization):
            assert (
                "static_normalizer.pkl" in created_files
            ), f"Expected the static normalizer to be saved. Not found in: {pformat(created_files)}"

        if cfg.dynamic_normalization:
            assert (
                "normalizer.pkl" in created_files
            ), f"Expected the normalizer to be saved. Not found in: {pformat(created_files)}"
        
        # check the predictions saved to netcdf
        assert (
            len([f for f in test_dir.glob("*.nc")]) > 0
        ), "Output NetCDF not saved to disk!"

    def test_linear_example(self):
        ds = create_linear_ds(epsilon_sigma=10)
        static_data = create_static_example_data(ds)

        cfg = Config(Path("tests/testconfigs/test_config.yml"))
        cfg._cfg["static_inputs"] = ["static_const", "static_rand"]
        cfg._cfg["seq_length"] = 2

        #  Train
        trainer = Trainer(cfg, ds, static_data=static_data)
        self.check_loaded_data(
            cfg,
            trainer,
            data=ds.sel(time=slice(cfg.train_start_date, cfg.train_end_date)),
        )
        losses = trainer.train_and_validate()
        save_loss_curves(losses, cfg)

        # Test
        tester = Tester(cfg, ds, static_data=static_data)
        preds = tester.run_test()
        for _ in range(2):
            save_timeseries(preds, cfg)

    def test_kenya_vci_example(self, tmp_path):
        cfg = Config(Path("tests/testconfigs/config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)

        cfg._cfg["data_path"] = Path("data/kenya.nc")
        cfg._cfg["n_epochs"] = 3

        ds, static = load_data(cfg)

        trainer = Trainer(cfg, ds, static_data=static)
        self.check_loaded_data(
            cfg,
            trainer,
            data=ds.sel(time=slice(cfg.train_start_date, cfg.train_end_date)),
        )

        losses = trainer.train_and_validate()

        tester = Tester(cfg, ds, static_data=static)
        preds = tester.run_test()

        return losses, preds

    def test_runoff_example(self, tmp_path):
        cfg = Config(Path("tests/testconfigs/config_runoff.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)

        cfg._cfg["data_path"] = Path("data/ALL_dynamic_ds.nc")
        cfg._cfg["static_data_path"] = Path("data/camels_static.nc")
        cfg._cfg["static_inputs"] = ["p_mean", "pet_mean", "area", "gauge_elev"]
        cfg._cfg["n_epochs"] = 3
        cfg._cfg["intial_forget_bias"] = 3
        cfg._cfg["clip_gradient_norm"] = 1
        cfg._cfg["loss"] = "NSE"
        cfg._cfg["dynamic_normalization"] = False
        cfg._cfg["static_normalization"] = True
        cfg._cfg["seq_length"] = 2
        cfg._cfg["input_variables"] = ['temperature', 'precipitation', "shortwave_rad"]
        cfg._cfg["model"] = "bilstm"


        ds, static = load_data(cfg)

        #  select subset of 3 basins
        basins = [1001, 2001, 2002]
        ds = ds.sel(station_id=basins)
        static = static.sel(station_id=basins)

        trainer = Trainer(cfg, ds, static_data=static)
        self.check_loaded_data(
            cfg,
            trainer,
            data=ds.sel(time=slice(cfg.train_start_date, cfg.train_end_date)),
        )

        losses = trainer.train_and_validate()

        tester = Tester(cfg, ds, static_data=static)
        preds = tester.run_test()

        return losses, preds


if __name__ == "__main__":
    #  create a random test dir (to check things are saved)
    tmp_path = create_test_dir()

    print(f"--- Writing to: {tmp_path} ---")

    t = TestPipeline()
    # t.test_linear_example()
    # losses, preds = t.test_kenya_vci_example()
    losses, preds = t.test_runoff_example(tmp_path)

    # #  plot the outputs
    # import matplotlib.pyplot as plt

    # f, ax = plt.subplots()
    # ax.plot(losses[0], label="Train")
    # ax.plot(losses[1], label="Validation")
    # ax.set_ylabel("MSE Loss")
    # ax.set_xlabel("Epoch")
    # plt.legend()

    t.check_output_files(tmp_path)
