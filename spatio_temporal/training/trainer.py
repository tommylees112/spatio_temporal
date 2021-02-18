import torch
import numpy as np
import xarray as xr

from spatio_temporal.training.base_trainer import BaseTrainer
from spatio_temporal.data.dataloader import PixelDataLoader
from spatio_temporal.config import Config


class Trainer(BaseTrainer):
    def __init__(self, cfg: Config, ds: xr.Dataset):
        super().__init__(cfg=cfg)

        # set / use the run directory
        self._create_folder_structure()
        self.device = cfg.device

        #  set random seeds
        self._set_random_seeds()
        # dump config file
        self.cfg.dump_config(self.cfg.run_dir)

        #  initialise dataloaders
        #  initialise model
        # train epochs

    def initialise_data(self, ds, mode: str = "train") -> None:
        """Load data from DataLoaders and store as attributes

        Args:
            ds ([type]): [description]
            mode (str, optional): [description]. Defaults to "train".

        Attributes:
            if mode == "train"
                self.train_dl ([PixelDataLoader]):
                self.valid_dl ([PixelDataLoader]):
            elif mode == "test"
                self.test_dl ([PixelDataLoader]):
        """
        # Train test split
        if mode == "train":
            # train period
            train_ds = ds[self.cfg.input_variables + [self.cfg.target_variable]].sel(
                time=slice(self.cfg.train_start_date, self.cfg.train_end_date)
            )
            self.train_dl = PixelDataLoader(train_ds, cfg=self.cfg, mode="train")

            #  validation period
            valid_ds = ds[self.cfg.input_variables + [self.cfg.target_variable]].sel(
                time=slice(self.cfg.validation_start_date, self.cfg.validation_end_date)
            )
            self.valid_dl = PixelDataLoader(valid_ds, cfg=self.cfg, mode="validation")

            self.normalizer = self.train_dl.normalizer
        else:
            # test period
            test_ds = ds[self.cfg.input_variables + [self.cfg.target_variable]].sel(
                time=slice(self.cfg.test_start_date, self.cfg.test_end_date)
            )
            self.test_dl = PixelDataLoader(
                test_ds, cfg=self.cfg, mode="test", normalizer=self.normalizer
            )
