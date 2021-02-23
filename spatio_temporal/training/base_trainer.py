from datetime import datetime
from typing import Optional, Union, List
import random
import pprint
import numpy as np
from pathlib import Path
import torch
from spatio_temporal.config import Config


class BaseTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.n_epochs = self.cfg.n_epochs

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

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(":")[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
                else:
                    self.device = torch.device(self.cfg.device)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        return pprint.pformat(self.cfg._cfg)

    @staticmethod
    def _set_seeds(cfg: Config):
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    #################################################
    ##############TRAINING GOODIES###################
    #################################################

    def _adjust_learning_rate(self, new_lr: float):
        # TODO: adjust the learning rate as go through
        # TODO: check pytorch implementations
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
