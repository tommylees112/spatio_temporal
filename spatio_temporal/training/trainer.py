import torch
import numpy as np
import xarray as xr
import torch.nn as nn
from tqdm import tqdm

#  library imports
from spatio_temporal.model.losses import RMSELoss
from spatio_temporal.training.base_trainer import BaseTrainer
from spatio_temporal.data.dataloader import PixelDataLoader
from spatio_temporal.config import Config
from spatio_temporal.model.lstm import LSTM
from spatio_temporal.model.linear_regression import LinearRegression


class Trainer(BaseTrainer):
    def __init__(self, cfg: Config, ds: xr.Dataset):
        super().__init__(cfg=cfg)

        # set / use the run directory
        self._create_folder_structure()
        self.device = self.cfg.device

        #  set random seeds
        self._set_seeds(self.cfg)
        
        # dump config file
        self.cfg.dump_config(self.cfg.run_dir)

        #  initialise dataloaders:: self.train_dl; self.valid_dl
        self.initialise_data(ds)

        #  initialise model:: self.model
        self.initialise_model()

        # intialise optimzier:: self.optimizer
        self._get_optimizer()

        # intialise loss:: self.loss_fn
        self._get_loss_obj()

        # train epochs

    #################################################
    ##############INITIALISATION#####################
    #################################################
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

    def initialise_model(self) -> None:
        #  TODO: def get_model from lookup: Dict[str, Model]
        self.model = LSTM(
            input_size=self.input_size,
            hidden_size=self.cfg.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.cfg.horizon,
        ).to(self.cfg.device)

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
        #  TODO: only initialise the necessary dataloaders
        # Train test split
        # if mode == "train":
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

        # else:
        # test period
        test_ds = ds[self.cfg.input_variables + [self.cfg.target_variable]].sel(
            time=slice(self.cfg.test_start_date, self.cfg.test_end_date)
        )
        self.test_dl = PixelDataLoader(
            test_ds, cfg=self.cfg, mode="test", normalizer=self.normalizer
        )

        #
        self.input_size = self.train_dl.input_size
        self.output_size = self.train_dl.output_size

    #################################################
    ##############TRAINING LOOP######################
    #################################################
    def train_and_validate(
        self,
    ):
        # cfg: Config, train_dl: PixelDataLoader, valid_dl: PixelDataLoader, model
        train_losses_all = []
        valid_losses_all = []

        for epoch in range(1, self.cfg.n_epochs + 1):
            self.model.train()
            train_loss = []

            #  batch the training data
            pbar = tqdm(self.train_dl, desc=f"Training Epoch {epoch}: ")
            for data in pbar:
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
                    assert False

                # backward pass (get gradients, step optimizer, delete old gradients)
                loss.backward()
                self.optimizer.step()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_gradient_norm
                    )

                # memorize the training loss
                train_loss.append(loss.item())
                pbar.set_postfix_str(f"{loss.item():.2f}")

            epoch_train_loss = np.mean(train_loss)

            # Save epoch weights
            self._save_epoch_information(epoch)

            #  TODO: early stopping
            # batch the validation data each epoch
            val_pbar = tqdm(self.valid_dl, desc=f"Validation Epoch {epoch}: ")
            with torch.no_grad():
                valid_loss = []
                for data in val_pbar:
                    x_val, y_val = data["x_d"], data["y"]
                    y_hat_val = self.model(x_val)
                    val_loss = self.loss_fn(y_hat_val["y_hat"], y_val)

                    # if torch.isnan(val_loss):
                    #     assert False, "Why is this happening?"
                    valid_loss.append(val_loss.item())

            epoch_valid_loss = np.mean(valid_loss)
            print(f"Train Loss: {np.sqrt(epoch_train_loss):.2f}")
            print(f"Valid Loss: {np.sqrt(epoch_valid_loss):.2f}")

            train_losses_all.append(epoch_train_loss)
            valid_losses_all.append(epoch_valid_loss)

        return np.array(train_losses_all), np.array(valid_losses_all)

    #################################################
    ##############TRAINING GOODIES###################
    #################################################

    def _adjust_learning_rate(self, new_lr: float):
        # TODO: adjust the learning rate as go through
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            param_group["lr"] = new_lr

    def _save_epoch_information(self, epoch: int) -> None:
        # SAVE model weights
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), str(weight_path))

        # SAVE optimizer state dict
        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))
