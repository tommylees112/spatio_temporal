import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random
import socket
from tqdm import tqdm
from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.model.linear_regression import LinearRegression
from spatio_temporal.training.train_utils import _to_device


def get_save_dir() -> Path:
    if socket.gethostname() == "GPU_MachineLearning":
        save_dir = Path("/home/tommy/spatio_temporal/runs/")
    else:
        save_dir = Path("/Users/tommylees/Downloads/")
    return save_dir


if __name__ == "__main__":
    ds = xr.open_dataset(Path("data/ALL_dynamic_ds.nc"))
    # ds = ds.isel(station_id=slice(0, 10))
    # cfg = Config(Path("tests/testconfigs/config_runoff.yml"))
    cfg = Config(Path("configs/runoff.yml"))
    cfg._cfg["scheduler"] = "step"
    trainer = Trainer(cfg, ds)

    #  overfit on one epoch
    epochs = 100
    model = trainer.model
    optimizer = trainer.optimizer
    loss_fn = trainer.loss_fn
    dl = trainer.train_dl
    scheduler = trainer.scheduler

    losses = []
    data = dl.__iter__().__next__()
    data = _to_device(data, cfg.device)
    x = data["x_d"]
    y = data["y"]

    for epoch in tqdm(np.arange(epochs)):
        optimizer.zero_grad()
        y_hat = model.forward(x)["y_hat"]
        loss = loss_fn(y_hat, y)
        loss.backward()

        optimizer.step()
        scheduler.step()

        losses.append(loss.detach())

    plt.plot(losses, marker="x")
    save_dir = get_save_dir()
    plt.gcf().savefig(save_dir / f"{random.random() * 10 :.0f}_plot.png")
    assert False
