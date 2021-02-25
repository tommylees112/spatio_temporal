from pathlib import Path
import xarray as xr
import pickle
import argparse

#  library imports
from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.training.tester import Tester
from tests.utils import (
    create_linear_ds,
    _test_sklearn_model,
    get_pollution_data_beijing,
)
from spatio_temporal.training.eval_utils import save_loss_curves, save_losses


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--baseline", type=bool, default=False)
    parser.add_argument("--run_dir", type=str)

    # parse args from user input
    args = vars(parser.parse_args())

    if (args["mode"] == "evaluate") and (args["run_dir"] is None):
        raise ValueError("Missing path to run directory")

    return args


if __name__ == "__main__":
    args = _get_args()
    mode = args["mode"]
    baseline = args["baseline"]

    #  load data
    #  TODO: automate the loading of data from the Config object somehow
    data_dir = Path("data")
    ds = pickle.load((data_dir / "kenya.pkl").open("rb"))
    # ds = xr.open_dataset(data_dir / "kenya.pkl")
    # ds = ds.isel(lat=slice(0, 10), lon=slice(0, 10))
    # ds = create_linear_ds().isel(lat=slice(0, 5), lon=slice(0, 5))
    # ds = xr.open_dataset(data_dir / "ALL_dynamic_ds.nc")
    # ds = ds.isel(station_id=slice(0, 10))
    # test_ds = ds.sel(station_id = 47001)
    # ds = get_pollution_data_beijing().to_xarray()

    #  Run Training and Evaluation
    if mode == "train":
        config_file = Path(args["config_file"])
        assert config_file.exists(), f"Expect config file at {config_file}"

        cfg = Config(cfg_path=config_file)

        # Train test split
        expt_class = trainer = Trainer(cfg, ds)
        tester = Tester(cfg, ds)

        if baseline:
            print("Testing sklearn Linear Regression")
            train_dl = trainer.train_dl
            test_dl = tester.test_dl
            _test_sklearn_model(train_dl, test_dl, cfg)

    #  Run Evaluation only
    else:
        test_dir = Path(args["run_dir"])
        cfg = Config(cfg_path=test_dir / "config.yml")
        expt_class = tester = Tester(cfg, ds)

    print()
    print(expt_class)
    model = expt_class.model
    # summary(model, input_size=(cfg.seq_length, expt_class.input_size), batch_size=-1)
    print(model)
    print()

    if mode == "train":
        losses = trainer.train_and_validate()

        # save the loss curves
        save_loss_curves(losses, cfg)
        save_losses(losses, cfg)

        # run test after training
        tester.run_test()

    elif mode == "evaluate":
        # RUN TEST !
        tester.run_test()
