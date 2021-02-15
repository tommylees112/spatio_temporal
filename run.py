import pandas as pd
import numpy as np
import pickle
import xarray as xr
from pathlib import Path
from functools import partial
from typing import Dict, Tuple
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


if __name__ == "__main__":
    #  TODO: move as much of this as possible to the Trainer object!
    data_dir = Path("data")
    run_dir = Path("runs")

    ds = pickle.load((data_dir / "kenya.pkl").open("rb"))
    cfg = Config(cfg_path=Path("configs/config.yml"))
    cfg.run_dir = run_dir
    trainer = Trainer(cfg)

    #  Settings
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train test split
    train_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
        time=slice(cfg.train_start_date, cfg.train_end_date)
    )
    valid_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
        time=slice(cfg.validation_start_date, cfg.validation_end_date)
    )
    test_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
        time=slice(cfg.test_start_date, cfg.test_end_date)
    )

    # Get DataLoaders
    base_dl = partial(
        PixelDataLoader,
        **dict(
            cfg=cfg,
            target_variable=cfg.target_variable,
            input_variables=cfg.input_variables,
            pixel_dims=cfg.pixel_dims,
            num_workers=cfg.num_workers,
            seq_length=cfg.seq_length,
            batch_size=cfg.batch_size,
            autoregressive=cfg.autoregressive,
        ),
    )

    train_dl = base_dl(train_ds, mode="train")
    valid_dl = base_dl(valid_ds, mode="validation")
    test_dl = base_dl(test_ds, mode="test")
    normalizer = train_dl.normalizer

    model = LSTM(
        input_size=train_dl.input_size,
        hidden_size=cfg.hidden_size,
        output_size=train_dl.output_size,
    )

    if cfg.loss == "MSE":
        loss_fn = nn.MSELoss()

    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [pam for pam in model.parameters()], lr=cfg.learning_rate
        )

    assert False
    for epoch in range(cfg.n_epochs):
        train_loss = []
        #  batch the training data
        pbar = tqdm(train_dl, desc=f"Training Epoch {epoch}: ")
        for x, y in pbar:
            # forward pass
            y_hat = model(x)

            # measure loss
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

    #  RUN TEST !
