import pandas as pd
import numpy as np
import pickle
import xarray as xr
from pathlib import Path
from functools import partial
from typing import Dict, Tuple, Optional, Any, Union
from tqdm import tqdm

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


def _create_dict_data_coords(
    y_hat: Tensor, y: Tensor
) -> Dict[str, Tuple[str, np.ndarray]]:
    # a tuple of (dims, data) ready for xr.Dataset creation
    #  TODO: forecast horizon ... ?
    data = {}
    data["obs"] = (("pixel", "time"), y.view(1, -1).detach().numpy())
    data["sim"] = (("pixel", "time"), y_hat.view(1, -1).detach().numpy())
    return data


def convert_to_xarray(
    data: Dict[str, Union[Tensor, Any]], y_hat: Tensor, forecast_horizon: int
) -> xr.Dataset:
    # back convert to xarray object ...
    data_xr = _create_dict_data_coords(y_hat=y_hat["y_hat"], y=data["y"])
    index = int(data["meta"]["index"])
    times = (
        data["meta"]["target_time"].detach().numpy().astype("datetime64[ns]").squeeze()
    )
    times = times.reshape(-1) if times.ndim == 0 else times
    pixel, _ = test_dl.dataset.lookup_table[int(index)]

    ds = xr.Dataset(data_xr, coords={"time": times, "pixel": [pixel]})
    return ds


def unnormalize_ds(
    dataloader: PixelDataLoader, ds: xr.Dataset, cfg: Config
) -> xr.Dataset:
    pixel = str(ds.pixel.values[0])
    unnorm = test_dl.normalizer.individual_inverse(
        ds, pixel_id=pixel, variable=cfg.target_variable
    )
    return unnorm


def get_final_xarray(
    data: Dict[str, Union[Tensor, Any]],
    y_hat: Tensor,
    dataloader: PixelDataLoader,
    cfg: Config,
) -> xr.Dataset:
    ds = convert_to_xarray(data=data, y_hat=y_hat, forecast_horizon=cfg.horizon)
    #  unnormalize the data (output scale)
    ds = unnormalize_ds(dataloader=test_dl, ds=ds, cfg=cfg)
    #  correct the formatting
    if "sample" in ds.coords:
        ds = ds.drop("sample")
    return ds


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

            #  get the final prediction
            _ds = get_final_xarray(data=data, y_hat=y_hat, dataloader=test_dl, cfg=cfg)
            all_ds_predictions.append(_ds)

    print("... Merging all predictions to one xr.Dataset")
    # merge and save results
    preds = xr.merge(all_ds_predictions)
    preds.to_netcdf(cfg.run_dir / f"test_predictions_E{str(epoch).zfill(3)}.nc")


def _get_loss_obj(cfg: Config):
    if cfg.loss == "MSE":
        loss_fn = nn.MSELoss()
    if cfg.loss == "RMSE":
        loss_fn = RMSELoss()

    return loss_fn


def _get_optimizer(cfg: Config):
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [pam for pam in model.parameters()], lr=cfg.learning_rate
        )
    return optimizer


def _adjust_learning_rate(optimizer, new_lr: float):
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

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss = []

        #  batch the training data
        pbar = tqdm(train_dl, desc=f"Training Epoch {epoch}: ")
        for data in pbar:
            x, y = data["x_d"], data["y"]

            # forward pass
            y_hat = model(x)

            # measure loss on forecasts
            loss = loss_fn(y_hat["y_hat"], y)

            # backward pass (get gradients, step optimizer, delete old gradients)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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
                val_loss = loss_fn(y_hat_val["y_hat"][-1], y_val[-1])
                valid_loss.append(val_loss.item())

        epoch_valid_loss = np.mean(valid_loss)
        print(f"Train Loss: {np.sqrt(epoch_train_loss):.2f}")
        print(f"Valid Loss: {np.sqrt(epoch_valid_loss):.2f}")


if __name__ == "__main__":
    TRAIN = False
    data_dir = Path("data")
    run_dir = Path("runs")

    ds = pickle.load((data_dir / "kenya.pkl").open("rb"))

    if TRAIN:
        cfg = Config(cfg_path=Path("configs/config.yml"))
        cfg.run_dir = run_dir
        trainer = Trainer(cfg, ds)

        # Train test split
        train_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
            time=slice(cfg.train_start_date, cfg.train_end_date)
        )
        valid_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
            time=slice(cfg.validation_start_date, cfg.validation_end_date)
        )

        # Get DataLoaders
        dl = train_dl = PixelDataLoader(train_ds, cfg=cfg, mode="train")
        valid_dl = PixelDataLoader(valid_ds, cfg=cfg, mode="validation")

    else:
        #  tester = Tester(cfg, ds)
        test_dir = Path("runs/test_kenya_1702_144610")
        cfg = Config(cfg_path=test_dir / "config.yml")

    test_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
        time=slice(cfg.test_start_date, cfg.test_end_date)
    )

    # Get DataLoaders
    dl = test_dl = PixelDataLoader(test_ds, cfg=cfg, mode="test")

    # Settings
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # TODO: def get_model from lookup: Dict[str, Model]
    # model = LSTM(
    #     input_size=dl.input_size,
    #     hidden_size=cfg.hidden_size,
    #     output_size=dl.output_size,
    #     forecast_horizon=dl.horizon,
    # ).to(cfg.device)
    model = LinearRegression(
        input_size=dl.input_size * cfg.seq_length,
        output_size=dl.output_size,
        forecast_horizon=cfg.horizon,
    ).to(cfg.device)

    print("-- Working with model: --")
    print(model)
    print()

    if TRAIN:
        train_and_validate(cfg, train_dl, valid_dl, model)
        run_test(cfg, test_dl, model)
    else:
        # RUN TEST !
        run_test(cfg, test_dl, model)
