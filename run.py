import pandas as pd
import numpy as np
import pickle
import xarray as xr
from pathlib import Path
from functools import partial
from typing import Dict, Tuple, Optional, Any, Union
from tqdm import tqdm
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

# pytorch imports
import torch
from torch import Tensor
from torch.optim import Adam
import torch.functional as F
import torch.nn as nn

#  library imports
from spatio_temporal.data.dataloader import PixelDataLoader
from spatio_temporal.model.lstm import LSTM
from spatio_temporal.model.linear_regression import LinearRegression
from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.model.losses import RMSELoss
from spatio_temporal.data.data_utils import load_all_data_from_dl_into_memory, _reshape
from tests.utils import create_linear_ds, create_dummy_vci_ds, _test_sklearn_model
from spatio_temporal.training.eval_utils import (
    _create_dict_data_coords_for_individual_sample,
    convert_individual_to_xarray,
    unnormalize_ds,
    get_individual_prediction_xarray_data,
    data_in_memory_to_xarray,
    scatter_plot,
    get_lists_of_metadata,
    plot_loss_curves,
    save_losses,
)


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


def _get_weight_file(cfg: Config, epoch: Optional[int] = None) -> Path:
    """Get file path to weight file"""
    if epoch is None:
        weight_file = sorted(list(cfg.run_dir.glob("model_epoch*.pt")))[-1]
    else:
        weight_file = cfg.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"

    return weight_file


def run_test(cfg: Config, test_dl: PixelDataLoader, model):
    weight_file = _get_weight_file(cfg)
    epoch = int(weight_file.name.split(".")[0][-3:])
    model.load_state_dict(torch.load(weight_file, map_location=cfg.device))

    all_ds_predictions = []
    model.eval()

    out = defaultdict(list)
    with torch.no_grad():
        test_pbar = tqdm(test_dl, desc=f"Test set Forward Pass: ")
        for data in test_pbar:
            x, y = data["x_d"], data["y"]
            y_hat = model(x)

            out["sim"].append(y_hat["y_hat"].detach().cpu().numpy())
            out["obs"].append(y.detach().cpu().numpy())
            pixels, times = get_lists_of_metadata(data, test_dl)
            out["time"].append(times)
            out["pixel"].append(pixels)

    return_dict: Dict[str, np.ndarray] = {}
    for key in out.keys():
        # concatenate over batch dimension (dimension = 0)
        var_ = np.concatenate(out[key])
        var_ = var_.squeeze() if var_.ndim == 3 else var_
        var_ = _reshape(var_)
        return_dict[key] = var_.flatten() if var_.shape[-1] == 1 else var_

    #  TODO: deal with multiple target variables (fh > 1)
    _preds = pd.DataFrame(return_dict).set_index(["pixel", "time"])
    preds = _preds.to_xarray()
    scatter_plot(preds, cfg, model="nn")

    preds.to_netcdf(cfg.run_dir / f"test_predictions_E{str(epoch).zfill(3)}.nc")


if __name__ == "__main__":
    #  TODO: linear model still has errors when fh = 0
    args = _get_args()
    mode = args["mode"]
    baseline = args["baseline"]

    #  load data
    data_dir = Path("data")
    # ds = pickle.load((data_dir / "kenya.pkl").open("rb"))
    # ds = ds.isel(lat=slice(0, 10), lon=slice(0, 10))
    ds = create_linear_ds().isel(lat=slice(0, 5), lon=slice(0, 5))
    # ds = xr.open_dataset(data_dir / "ALL_dynamic_ds.nc")
    # ds = ds.isel(station_id=slice(0, 10))

    #  Run Training and Evaluation
    if mode == "train":
        config_file = Path(args["config_file"])
        assert config_file.exists(), f"Expect config file at {config_file}"

        cfg = Config(cfg_path=config_file)

    #  Run Evaluation only
    else:
        #  tester = Tester(cfg, ds)
        test_dir = Path(args["run_dir"])
        cfg = Config(cfg_path=test_dir / "config.yml")

    # Train test split
    trainer = Trainer(cfg, ds)
    dl = train_dl = trainer.train_dl
    valid_dl = trainer.valid_dl
    test_dl = trainer.test_dl

    # TODO: def get_model from lookup: Dict[str, Model]
    model = trainer.model

    if baseline:
        print("Testing sklearn Linear Regression")
        _test_sklearn_model(train_dl, test_dl, cfg)

    print("-- Working with model: --")
    print(model)
    print()

    if mode == "train":
        losses = train_and_validate(cfg, train_dl, valid_dl, model)
        run_test(cfg, test_dl, model)

        # save the loss curves
        plot_loss_curves(losses, cfg)
        save_losses(losses, cfg)

    elif mode == "evaluate":
        # RUN TEST !
        run_test(cfg, test_dl, model)
