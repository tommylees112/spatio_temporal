from pathlib import Path
import xarray as xr
from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.training.tester import Tester
from spatio_temporal.training.eval_utils import (
    _plot_loss_curves,
    save_loss_curves,
    save_timeseries,
)


if __name__ == "__main__":
    #  LOAD IN DATA
    ds = xr.open_dataset("data/data_india_regions.nc").sortby("time")
    cfg = Config(Path("configs/india_region.yml"))
    cfg._cfg["n_epochs"] = 150
    trainer = Trainer(cfg, ds)

    #  TRAIN
    losses = trainer.train_and_validate()
    save_loss_curves(losses, cfg)

    #  TEST
    tester = Tester(cfg, ds)
    preds = tester.run_test(unnormalize=True)
    for _ in range(2):
        save_timeseries(preds, cfg)
