import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import torch
from spatio_temporal.data.dataloader import PixelDataLoader
from spatio_temporal.model.lstm import LSTM
from spatio_temporal.config import Config
from tests.utils import (
    _make_dataset,
    get_oxford_weather_data,
    create_and_assign_temp_run_path_to_config,
)
from torch.utils.data import random_split
from torch.nn import functional as F
from tqdm import tqdm


class TestModels:
    def test_lstm_forward_pass(self, tmp_path):
        ds = pickle.load(Path("data/kenya.pkl").open("rb"))
        cfg = Config(Path("tests/testconfigs/config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)
        dl = PixelDataLoader(ds, cfg=cfg, mode="train")

        model = LSTM(
            input_size=dl.input_size,
            hidden_size=cfg.hidden_size,
            output_size=dl.output_size,
        )
        data = dl.__iter__().__next__()
        x, y = data["x_d"], data["y"]
        y_hat = model(x)

        assert all(np.isin(["h_n", "c_n", "y_hat"], [k for k in y_hat.keys()]))

    def test_single_train(self, tmp_path):
        torch.manual_seed(1)
        np.random.seed(1)

        batch_size = 30
        seq_length = 10
        input_variables = ["precip", "t2m", "SMsurf"]
        hidden_size = 64
        ds = pickle.load(Path("data/kenya.pkl").open("rb"))
        cfg = Config(Path("tests/testconfigs/config.yml"))
        create_and_assign_temp_run_path_to_config(cfg, tmp_path)

        dl = PixelDataLoader(
            ds, mode="train", cfg=cfg, num_workers=1, batch_size=batch_size,
        )

        data = dl.__iter__().__next__()
        x, y = data["x_d"], data["y"]

        # are we working with batches or individual predictions?
        x = x.unsqueeze(0) if x.ndim == 2 else x

        model = (
            LSTM(
                input_size=dl.input_size,
                hidden_size=hidden_size,
                output_size=dl.output_size,
            )
            .float()
            .to(cfg.device)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_obj = F.mse_loss
        before = model.forward(x)
        for data in tqdm(dl):
            input, target = data["x_d"], data["y"]
            optimizer.zero_grad()
            yhat = model.forward(input)
            loss = loss_obj(yhat["y_hat"], target)
            loss.backward()
            optimizer.step()
            break

        after = model.forward(x)

        loss_bf = loss_obj(before["y_hat"], y)
        loss_af = loss_obj(after["y_hat"], y)

        # NOTE: the LSTM only returns the final hidden and cell state layer NOT each timestep
        #  TODO: why is the LSTM returning a hidden array of shape (seq_length, 1, hs)
        assert before["h_n"].shape == (seq_length, 1, hidden_size)
        assert before["y_hat"].shape == (batch_size, seq_length, 1)

        assert (
            loss_af < loss_bf
        ), "The model did not learn anything after one epoch of training"
