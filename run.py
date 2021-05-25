from pathlib import Path
import xarray as xr
import pickle
import argparse
from typing import Union, Tuple, Optional

#  library imports
from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.training.tester import Tester
from tests.utils import (
    create_linear_ds,
    _test_sklearn_model,
    get_pollution_data_beijing,
)
from spatio_temporal.training.eval_utils import (
    save_loss_curves,
    save_losses,
    save_timeseries,
)


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--baseline", type=bool, default=False)
    parser.add_argument("--overfit_test", type=bool, default=True)
    parser.add_argument("--run_dir", type=str)

    # parse args from user input
    args = vars(parser.parse_args())

    if (args["mode"] == "evaluate") and (args["run_dir"] is None):
        raise ValueError("Missing path to run directory")

    return args


def load_data(cfg: Config) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
    ds = xr.open_dataset(cfg.data_path)
    if cfg.static_data_path is not None:
        static_data = xr.open_dataset(cfg.static_data_path)
    else:
        static_data = None

    return ds, static_data


if __name__ == "__main__":
    args = _get_args()
    mode = args["mode"]
    baseline = args["baseline"]
    overfit_test = args["overfit_test"]

    #  load data
    #  TODO: automate the loading of data from the Config object somehow
    data_dir = Path("data")

    ## kenya experiments
    # ds = pickle.load((data_dir / "kenya.pkl").open("rb"))
    # ds = xr.open_dataset(data_dir / "kenya.pkl")
    # ds = ds.isel(lat=slice(0, 10), lon=slice(0, 10))

    ## discharge experiments
    #  ds = xr.open_dataset(data_dir / "ALL_dynamic_ds.nc")
    #  ds = ds.isel(station_id=slice(0, 10))

    ## test experiments
    # ds = create_linear_ds().isel(lat=slice(0, 5), lon=slice(0, 5))
    # test_ds = ds.sel(station_id = 47001)
    # ds = get_pollution_data_beijing().to_xarray()

    ## india experiments
    # ds = xr.open_dataset("data/data_india_regions.nc").sortby("time")
    # ds = xr.open_dataset("data/data_india_full.nc").sortby("time")

    ## river level data
    # ds = xr.open_dataset("data/camels_river_level_data.nc")

    #  Run Training and Evaluation
    expt_class: Union[Trainer, Tester] 
    if mode == "train":
        config_file = Path(args["config_file"])
        assert config_file.exists(), f"Expect config file at {config_file}"

        cfg = Config(cfg_path=config_file)

        # Load in data
        ds, static_data = load_data(cfg)

        # Train test split
        expt_class = trainer = Trainer(cfg, ds, static_data=static_data)
        tester = Tester(cfg, ds, static_data=static_data)

        if overfit_test:
            #  run test on training data to check for overfitting
            overfitting_tester = Tester(cfg, ds, subset="train")

        if baseline:
            print("Testing sklearn Linear Regression")
            train_dl = trainer.train_dl
            test_dl = tester.test_dl
            _test_sklearn_model(train_dl, test_dl, cfg)

    #  Run Evaluation only
    else:
        test_dir = Path(args["run_dir"])
        cfg = Config(cfg_path=test_dir / "config.yml")

        # Load in data
        ds, static_data = load_data(cfg)
        expt_class = tester = Tester(cfg, ds, static_data=static_data)

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
        preds = tester.run_test()
        save_timeseries(preds, cfg=cfg, n=2)

        if overfit_test:
            overfitting_tester.run_test()

    elif mode == "evaluate":
        # RUN TEST !
        tester.run_test()
