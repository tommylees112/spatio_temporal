from pathlib import Path
import xarray as xr
from spatio_temporal.config import Config
from spatio_temporal.training.tester import Tester
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.data.data_utils import initialize_normalizer
from spatio_temporal.training.eval_utils import (
    save_loss_curves,
    save_losses,
    save_timeseries,
)


if __name__ == "__main__":
    OVERFIT = False

    data_dir = Path("data/")

    # load dataset
    ds_path = data_dir / "ALL_dynamic_ds.nc"
    ds = xr.open_dataset(ds_path)
    ds = ds.isel(station_id=slice(0, 10))

    # load config
    cfg_path = Path("tests/testconfigs/config_runoff.yml")
    cfg = Config(cfg_path=cfg_path)
    cfg._cfg["experiment_name"] = "01_test_discharge_TL"

    # Train Test Split
    trainer = Trainer(cfg, ds)
    tester = Tester(cfg, ds)

    # Overfit on train data
    if OVERFIT:
        overfitting_tester = Tester(cfg, ds, subset="train")
        overfitting_tester.run_test()

    # train-test loop
    losses = trainer.train_and_validate()

    # save the loss curves
    save_loss_curves(losses, cfg)
    save_losses(losses, cfg)

    # run test after training
    preds = tester.run_test()
    save_timeseries(preds, cfg=cfg, n=2)

    assert False
