import torch
import numpy as np
import xarray as xr
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Union, Optional
import torch.optim as optim

#  library imports
from spatio_temporal.model.losses import RMSELoss
from spatio_temporal.training.base_trainer import BaseTrainer
from spatio_temporal.data.dataloader import PixelDataLoader
from spatio_temporal.config import Config
from spatio_temporal.model.lstm import LSTM
from spatio_temporal.model.linear_regression import LinearRegression
from spatio_temporal.data.data_utils import train_test_split
from spatio_temporal.training.train_utils import _to_device


class Memory:
    train_losses: Optional[Union[List[float], np.ndarray]] = None
    valid_losses: Optional[Union[List[float], np.ndarray]] = None

    def __post_init__(self):
        self.train_losses = []
        self.valid_losses = []

    def to_arrays(self):
        self.train_losses = np.array(self.train_losses)
        self.valid_losses = np.array(self.valid_losses)


class Trainer(BaseTrainer):
    def __init__(
        self, cfg: Config, ds: xr.Dataset, _allow_subsequent_nan_losses: int = 5
    ):
        super().__init__(cfg=cfg)

        # set / use the run directory
        self._create_folder_structure()
        self.device = self.cfg.device
        self._allow_subsequent_nan_losses = _allow_subsequent_nan_losses

        #  set random seeds
        self._set_seeds(self.cfg)

        # dump config file
        self.cfg.dump_config(self.cfg.run_dir)

        #  initialise dataloaders:: self.train_dl; self.valid_dl
        self.initialise_data(ds)
        self.input_size = self.train_dl.input_size
        self.output_size = self.train_dl.output_size

        # initialise normalizer
        self.normalizer = self.train_dl.normalizer

        #  initialise model:: self.model
        self.initialise_model()

        # intialise optimzier:: self.optimizer
        self._get_optimizer()

        # initialise scheduler:: self.scheduler
        self._get_scheduler()

        # intialise loss:: self.loss_fn
        self._get_loss_obj()

    #################################################
    ##############INITIALISATION#####################
    #################################################

    def initialise_memory(self) -> None:
        #  keys:: 'train_losses' 'valid_losses'
        self.memory = Memory()
        self.memory.__post_init__()

    def _get_loss_obj(self) -> None:
        if self.cfg.loss == "MSE":
            loss_fn = nn.MSELoss()
        if self.cfg.loss == "RMSE":
            loss_fn = RMSELoss()
        if self.cfg.loss == "huber":
            loss_fn = nn.SmoothL1Loss()

        self.loss_fn = loss_fn

    def _get_optimizer(self) -> None:
        if self.cfg.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                [pam for pam in self.model.parameters()], lr=self.cfg.learning_rate
            )
        self.optimizer = optimizer

    def _reset_scheduler(self) -> None:
        # TODO: cfg options for step_size and gamma
        if self.cfg.scheduler is not None:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )

    def _get_scheduler(self) -> None:
        # https://discuss.pytorch.org/t/how-to-implement-torch-optim-lr-scheduler-cosineannealinglr/28797/6
        self._reset_scheduler()

    def initialise_model(self) -> None:
        #  TODO: def get_model from lookup: Dict[str, Model]
        self.model = LSTM(
            input_size=self.input_size,
            hidden_size=self.cfg.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.cfg.horizon,
        ).to(self.cfg.device)

    def initialise_data(self, ds: xr.Dataset, mode: str = "train") -> None:
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
        #  TODO: only initialise the necessary dataloaders
        # Train-Validation split
        # train period
        train_ds = train_test_split(ds, cfg=self.cfg, subset="train")
        self.train_dl = PixelDataLoader(
            train_ds,
            cfg=self.cfg,
            mode="train",
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            batch_size=self.cfg.batch_size,
        )
        assert (
            self.train_dl.dataset.y != {}
        ), f"Train Period loads in no data for period {self.cfg.train_start_date} -- {self.cfg.train_end_date} with seq_length {self.cfg.seq_length}"

        #  validation period
        valid_ds = train_test_split(ds, cfg=self.cfg, subset="validation")
        self.valid_dl = PixelDataLoader(
            valid_ds,
            cfg=self.cfg,
            mode="validation",
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            batch_size=self.cfg.batch_size,
        )

    #################################################
    ##############TRAINING LOOP######################
    #################################################
    def _train_one_epoch(self, epoch: int) -> np.ndarray:
        self.model.train()
        train_loss = []
        nan_count = 0

        #  batch the training data and iterate over batches
        pbar = tqdm(self.train_dl, desc=f"Training Epoch {epoch}: ")
        for data in pbar:
            #  to GPU
            data = _to_device(data, self.device)

            x, y = data["x_d"], data["y"]

            #  zero gradient before forward pass
            self.optimizer.zero_grad()

            # forward pass
            y_hat = self.model(x)

            # measure loss on forecasts
            if not (y_hat["y_hat"].ndim == y.ndim):
                y = y.squeeze(0)
            loss = self.loss_fn(y_hat["y_hat"], y)

            if torch.isnan(loss):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(
                        f"Loss was NaN for {nan_count} times in a row. Stopped training."
                    )

            # backward pass (get gradients, step optimizer, delete old gradients)
            loss.backward()

            #  TODO: clip gradients after backward pass; before optimizer step
            if self.cfg.clip_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_gradient_norm
                )

            self.optimizer.step()
            if self.scheduler is not None:
                learning_rate = self.optimizer.param_groups[0]["lr"]
            else:
                learning_rate = self.cfg.learning_rate

            # memorize the training loss & set bar info
            pbar.set_postfix_str(f"{loss.item():.2f} -- LR {learning_rate:.2e}")
            train_loss.append(loss.item())

        epoch_train_loss = np.mean(train_loss)
        return epoch_train_loss

    def _validate_epoch(self, epoch: int) -> np.ndarray:
        #  TODO: early stopping
        # batch the validation data and run validation forward pass
        val_pbar = tqdm(self.valid_dl, desc=f"Validation Epoch {epoch}: ")
        with torch.no_grad():
            valid_loss = []
            for data in val_pbar:

                #  to GPU
                data = _to_device(data, self.device)

                # run forward pass
                x_val, y_val = data["x_d"], data["y"]
                y_hat_val = self.model(x_val)
                val_loss = self.loss_fn(y_hat_val["y_hat"], y_val)

                valid_loss.append(val_loss.item())

        epoch_valid_loss = np.mean(valid_loss)
        return epoch_valid_loss

    def train_and_validate(self):
        # store losses:: self.memory
        self.initialise_memory()

        for epoch in range(1, self.cfg.n_epochs + 1):
            epoch_train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()
            # self._reset_scheduler()

            # Save epoch weights
            self._save_epoch_information(epoch)

            if epoch % self.cfg.validate_every_n == 0:
                epoch_valid_loss = self._validate_epoch(epoch)
            else:
                epoch_valid_loss = np.nan

            print(f"Train Loss: {epoch_train_loss:.2f}")
            print(f"Valid Loss: {epoch_valid_loss:.2f}")

            self.memory.train_losses.append(epoch_train_loss)
            self.memory.valid_losses.append(epoch_valid_loss)

        self.memory.to_arrays()
        return self.memory.train_losses, self.memory.valid_losses
