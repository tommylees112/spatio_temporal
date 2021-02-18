import pandas as pd
import numpy as np
import pickle
import xarray as xr
from pathlib import Path
from functools import partial
from typing import Dict, Tuple, Optional, Any, Union
from tqdm import tqdm
import argparse
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
from spatio_temporal.data.data_utils import load_all_data_from_dl_into_memory
from tests.utils import create_linear_ds, create_dummy_vci_ds, _test_sklearn_model
from spatio_temporal.training.eval_utils import (
    _create_dict_data_coords_for_individual_sample,
    convert_individual_to_xarray,
    unnormalize_ds,
    get_individual_prediction_xarray_data,
    data_in_memory_to_xarray,
    scatter_plot,
)


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--run_dir", type=str)

    # parse args from user input
    args = vars(parser.parse_args())

    if (args["mode"] == "evaluate") and (args["run_dir"] is None):
        raise ValueError("Missing path to run directory")

    return args


def _save_epoch_information(run_dir: Path, epoch: int, model, optimizer) -> None:
    # SAVE model weights
    weight_path = run_dir / f"model_epoch{epoch:03d}.pt"
    torch.save(model.state_dict(), str(weight_path))

    # SAVE optimizer state dict
    optimizer_path = run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
    torch.save(optimizer.state_dict(), str(optimizer_path))


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
    with torch.no_grad():
        test_pbar = tqdm(test_dl, desc=f"Test set Forward Pass: ")
        for data in test_pbar:
            x, y = data["x_d"], data["y"]
            y_hat = model(x)

            # TODO: deal with multiple forecast horizons
            # (y_hat["y_hat"].squeeze().shape[0] > 1)
            #  get the final prediction
            _ds = get_individual_prediction_xarray_data(
                data=data, y_hat=y_hat, dataloader=test_dl, cfg=cfg
            )
            all_ds_predictions.append(_ds)

    print("... Merging all predictions to one xr.Dataset")
    # merge and save results
    preds = xr.merge(all_ds_predictions)
    scatter_plot(preds, cfg, model="nn")

    preds.to_netcdf(cfg.run_dir / f"test_predictions_E{str(epoch).zfill(3)}.nc")


def _get_loss_obj(cfg: Config):
    if cfg.loss == "MSE":
        loss_fn = nn.MSELoss()
    if cfg.loss == "RMSE":
        loss_fn = RMSELoss()
    if cfg.loss == "huber":
        loss_fn = nn.SmoothL1Loss()

    return loss_fn


def _get_optimizer(cfg: Config):
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [pam for pam in model.parameters()], lr=cfg.learning_rate
        )
    return optimizer


def _adjust_learning_rate(optimizer, new_lr: float):
    # TODO: adjust the learning rate as go through
    for param_group in optimizer.param_groups:
        old_lr = param_group["lr"]
        param_group["lr"] = new_lr


def train_and_validate(
    cfg: Config, train_dl: PixelDataLoader, valid_dl: PixelDataLoader, model
):
    # TODO: move as much of this as possible to the Trainer object!
    # get loss & optimizer
    loss_fn = _get_loss_obj(cfg)
    optimizer = _get_optimizer(cfg)
    train_losses_all = []
    valid_losses_all = []

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss = []

        #  batch the training data
        pbar = tqdm(train_dl, desc=f"Training Epoch {epoch}: ")
        for data in pbar:
            x, y = data["x_d"], data["y"]

            #  zero gradient before forward pass
            optimizer.zero_grad()

            # forward pass
            y_hat = model(x)

            # measure loss on forecasts
            if not (y_hat["y_hat"].ndim == y.ndim):
                y = y.squeeze(0)
            loss = loss_fn(y_hat["y_hat"], y)

            if torch.isnan(loss):
                #  TODO: why nans with 0 horizon inputs
                assert False

            # backward pass (get gradients, step optimizer, delete old gradients)
            loss.backward()
            optimizer.step()

            if cfg.clip_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.clip_gradient_norm
                )

            # memorize the training loss
            train_loss.append(loss.item())
            pbar.set_postfix_str(f"{loss.item():.2f}")

        epoch_train_loss = np.mean(train_loss)

        # Save epoch weights
        _save_epoch_information(cfg.run_dir, epoch, model=model, optimizer=optimizer)

        #  TODO: early stopping
        # batch the validation data each epoch
        val_pbar = tqdm(valid_dl, desc=f"Validation Epoch {epoch}: ")
        with torch.no_grad():
            valid_loss = []
            for data in val_pbar:
                x_val, y_val = data["x_d"], data["y"]
                y_hat_val = model(x_val)
                val_loss = loss_fn(y_hat_val["y_hat"], y_val)

                # if torch.isnan(val_loss):
                #     assert False, "Why is this happening?"
                valid_loss.append(val_loss.item())

        epoch_valid_loss = np.mean(valid_loss)
        print(f"Train Loss: {np.sqrt(epoch_train_loss):.2f}")
        print(f"Valid Loss: {np.sqrt(epoch_valid_loss):.2f}")

        train_losses_all.append(epoch_train_loss)
        valid_losses_all.append(epoch_valid_loss)

    return np.array(train_losses_all), np.array(valid_losses_all)


if __name__ == "__main__":
    # TODO: linear model still has errors when fh = 0
    args = _get_args()
    mode = args["mode"]

    #  load data
    data_dir = Path("data")
    ds = pickle.load((data_dir / "kenya.pkl").open("rb"))
    # ds = ds.isel(lat=slice(0, 10), lon=slice(0, 10))
    # ds = create_linear_ds()

    #  Run Training and Evaluation
    if mode == "train":
        config_file = Path(args["config_file"])
        assert config_file.exists(), f"Expect config file at {config_file}"

        cfg = Config(cfg_path=config_file)
        trainer = Trainer(cfg, ds)
    #  Run Evaluation only
    else:
        #  tester = Tester(cfg, ds)
        test_dir = Path(args["run_dir"])
        cfg = Config(cfg_path=test_dir / "config.yml")

    # Train test split
    # Get DataLoaders
    train_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
        time=slice(cfg.train_start_date, cfg.train_end_date)
    )
    valid_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
        time=slice(cfg.validation_start_date, cfg.validation_end_date)
    )
    dl = train_dl = PixelDataLoader(
        train_ds, cfg=cfg, mode="train", batch_size=cfg.batch_size
    )
    valid_dl = PixelDataLoader(valid_ds, cfg=cfg, mode="validation")

    test_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
        time=slice(cfg.test_start_date, cfg.test_end_date)
    )
    dl = test_dl = PixelDataLoader(test_ds, cfg=cfg, mode="test")

    print("Testing sklearn Linear Regression")
    _test_sklearn_model(train_dl, test_dl, cfg)

    # TODO: def get_model from lookup: Dict[str, Model]
    model = LSTM(
        input_size=dl.input_size,
        hidden_size=cfg.hidden_size,
        output_size=dl.output_size,
        forecast_horizon=cfg.horizon,
    ).to(cfg.device)

    print("-- Working with model: --")
    print(model)
    print()

    if mode == "train":
        train_losses_all, valid_losses_all = train_and_validate(
            cfg, train_dl, valid_dl, model
        )
        run_test(cfg, test_dl, model)
    elif mode == "evaluate":
        # RUN TEST !
        run_test(cfg, test_dl, model)
