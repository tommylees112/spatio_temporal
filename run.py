import pandas as pd
import numpy as np
import pickle
import xarray as xr
from pathlib import Path
from functools import partial
from typing import Dict, Tuple, Optional
from tqdm import tqdm

# pytorch imports
import torch
from torch.optim import Adam
import torch.functional as F
import torch.nn as nn

#  library imports
from spatio_temporal.data.dataloader import PixelDataLoader
from spatio_temporal.model.lstm import LSTM
from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer


def _get_weight_file(cfg: Config, epoch: Optional[int] = None):
    """Get file path to weight file"""
    if epoch is None:
        weight_file = sorted(list(cfg.run_dir.glob("model_epoch*.pt")))[-1]
    else:
        weight_file = cfg.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"

    return weight_file


def run_test(cfg: Config, test_dl: PixelDataLoader):
    # RUN TEST !
    model = LSTM(
        input_size=test_dl.input_size,
        hidden_size=cfg.hidden_size,
        output_size=test_dl.output_size,
    )

    weight_file = _get_weight_file(cfg)
    model.load_state_dict(torch.load(weight_file, map_location=cfg.device))
    
    model.eval()
    with torch.no_grad():
        test_pbar = tqdm(test_dl, desc=f"Test set Forward Pass: ")
        for x, y in test_pbar:
            y_hat = model(x)
            # back convert to xarray object ...
            assert False
            
            # rescale predictions
            # test_dl.normalizer.individual_inverse(data, pixel_id=pixel_id, variable=cfg.target_variable)

            # create xarray object (sim, obs)
            # _create_xarray(y_hat, y)

            # save results
            # _save_results(results, epoch)


def train(cfg, train_dl, valid_dl, model):
    # TODO: move as much of this as possible to the Trainer object!
    # get loss & optimizer
    if cfg.loss == "MSE":
        loss_fn = nn.MSELoss()

    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [pam for pam in model.parameters()], lr=cfg.learning_rate
        )

    for epoch in range(cfg.n_epochs):
        train_loss = []
        #  batch the training data
        pbar = tqdm(train_dl, desc=f"Training Epoch {epoch}: ")
        for x, y in pbar:
            # forward pass
            y_hat = model(x)

            # measure loss on all (test memory) or only on last (test_forecast)
            loss = loss_fn(y_hat["y_hat"], y)

            # backward pass
            loss.backward()
            optimizer.step()

            # memorize the training loss
            train_loss.append(loss.item())
            pbar.set_postfix_str(f"{loss.item():.2f}")

        epoch_train_loss = np.mean(train_loss)

        # batch the validation data each epoch
        val_pbar = tqdm(valid_dl, desc=f"Validation Epoch {epoch}: ")
        with torch.no_grad():
            valid_loss = []
            for x_val, y_val in val_pbar:
                y_hat_val = model(x_val)
                val_loss = loss_fn(y_hat_val["y_hat"], y_val)
                valid_loss.append(val_loss.item())

        epoch_train_loss = np.mean(valid_loss)
        print(f"Valid Loss: {np.sqrt(np.mean(valid_loss)):.2f}")

        # SAVE epoch results
        weight_path = cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(model.state_dict(), str(weight_path))

        optimizer_path = cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(optimizer.state_dict(), str(optimizer_path))


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
        cfg = Config(cfg_path=Path("runs/test_1602_154323/config.yml"))

        test_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
            time=slice(cfg.test_start_date, cfg.test_end_date)
        )

        # Get DataLoaders
        dl = test_dl = PixelDataLoader(test_ds, cfg=cfg, mode="test")
    
    # Settings
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTM(
        input_size=dl.input_size,
        hidden_size=cfg.hidden_size,
        output_size=dl.output_size,
    )

    if TRAIN:
        train(cfg, train_dl, valid_dl, model)
    else:
        run_test(cfg, test_dl)

