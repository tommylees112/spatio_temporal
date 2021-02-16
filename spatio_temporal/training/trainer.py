from datetime import datetime
from pathlib import Path
import torch
from spatio_temporal.config import Config
import random
import numpy as np
from typing import Optional
import xarray as xr
# from spatio_temporal.training.base_trainer import BaseTrainer


class Trainer:
    def __init__(self, cfg: Config, ds: xr.Dataset):
        self.cfg = cfg
        # set / use the run directory
        self._create_folder_structure()
        self.model = None
        self.optimizer = None
        self.device = cfg.device

        #  set random seeds
        self._set_random_seeds()
        # dump config file
        self.cfg.dump_config(self.cfg.run_dir)

        #  initialise dataloaders
        # train epochs

    def initialise_data(ds, mode: str = "train"):
        # Train test split
        if mode == "train":
            # train period
            train_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
                time=slice(cfg.train_start_date, cfg.train_end_date)
            )
            self.train_dl = PixelDataLoader(train_ds, cfg=cfg, mode="train")

            #  validation period
            valid_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
                time=slice(cfg.validation_start_date, cfg.validation_end_date)
            )
            self.valid_dl = PixelDataLoader(valid_ds, cfg=cfg, mode="validation")

            self.normalizer = self.train_dl.normalizer
        else:
            # test period
            test_ds = ds[cfg.input_variables + [cfg.target_variable]].sel(
                time=slice(cfg.test_start_date, cfg.test_end_date)
            )
            self.test_dl = PixelDataLoader(
                test_ds, cfg=cfg, mode="test", normalizer=self.normalizer
            )

    def _set_random_seeds(self):
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.uniform(low=0, high=1e6))

        # fix random seeds for various packages
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _create_folder_structure(self):
        run_name = self._create_datetime_folder_name()
        if self.cfg.run_dir is None:
            self.cfg.run_dir = Path().cwd() / "runs" / run_name
        else:
            self.cfg.run_dir = self.cfg.run_dir / run_name

        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)

    def _create_datetime_folder_name(self) -> str:
        now = datetime.now()
        day = f"{now.day}".zfill(2)
        month = f"{now.month}".zfill(2)
        hour = f"{now.hour}".zfill(2)
        minute = f"{now.minute}".zfill(2)
        second = f"{now.second}".zfill(2)
        run_name = f"{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}"
        return run_name

    def _save_weights_and_optimizer(self, epoch: int):
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), str(weight_path))

        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

    def _get_weight_file(self, epoch: Optional[int]):
        """Get file path to weight file"""
        if epoch is None:
            weight_file = sorted(list(self.cfg.run_dir.glob("model_epoch*.pt")))[-1]
        else:
            weight_file = self.cfg.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"

        return weight_file

    def _load_weights(self, epoch: int = None):
        """Load weights of a certain (or the last) epoch into the model."""
        weight_file = self._get_weight_file(epoch)

        self.model.load_state_dict(torch.load(weight_file, map_location=self.device))
