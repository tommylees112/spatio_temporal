import torch
from pathlib import Path
from typing import Optional, Dict, Tuple, List, DefaultDict
import numpy as np
import xarray as xr
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from spatio_temporal.config import Config
from spatio_temporal.data.dataloader import PixelDataLoader
from spatio_temporal.training.eval_utils import (
    create_metadata_arrays,
    scatter_plot,
)
from spatio_temporal.data.data_utils import _reshape, train_test_split
from spatio_temporal.training.train_utils import _to_device, get_model


class Tester:
    def __init__(self, cfg: Config, ds: xr.Dataset, subset: str = "test"):
        self.cfg = cfg

        self.device = self.cfg.device

        # load test dataloader:: self.test_dl
        assert subset in ["test", "train", "validation"]
        self.subset = subset

        self.initialise_data(ds, subset=self.subset)
        self.dynamic_input_size = self.test_dl.dynamic_input_size
        self.static_input_size = self.test_dl.static_input_size
        self.forecast_input_size = self.test_dl.forecast_input_size
        self.input_size = (
            self.dynamic_input_size + self.static_input_size + self.forecast_input_size
        )
        self.output_size = self.test_dl.output_size

        # load model and model weights:: self.model
        self.load_model()

    def __repr__(self):
        return self.cfg._cfg.__repr__()

    def initialise_data(self, ds: xr.Dataset, subset: str = "test") -> None:

        test_ds = train_test_split(ds, cfg=self.cfg, subset=subset)
        #  NOTE: normalizer should be read from the cfg.run_dir directory
        self.test_dl = PixelDataLoader(
            test_ds,
            cfg=self.cfg,
            mode="test",
            normalizer=None,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )

        period_start = self.cfg._cfg[f"{subset}_start_date"]
        period_end = self.cfg._cfg[f"{subset}_end_date"]
        assert (
            self.test_dl.dataset.y != {}
        ), f"{subset} Period loads in no data for period {period_start} -- {period_end} with seq_length {self.cfg.seq_length}"

    def load_model(self):
        #  TODO: def get_model from lookup: Dict[str, Model]
        self.model = get_model(
            cfg=self.cfg, input_size=self.input_size, output_size=self.output_size,
        )

    @staticmethod
    def _get_weight_file(cfg: Config, epoch: Optional[int] = None) -> Path:
        """Get file path to weight file"""
        if epoch is None:
            weight_file = sorted(list(cfg.run_dir.glob("model_epoch*.pt")))[-1]
        else:
            weight_file = cfg.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"

        return weight_file

    @staticmethod
    def _unpack_obs_sim_time(
        return_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        # deal with multiple target variables (fh > 1)
        target_horizons = return_dict["obs"].shape[-1]
        if (target_horizons > 1) & (return_dict["obs"].ndim > 1):
            obs = [return_dict["obs"][:, i] for i in range(target_horizons)]
            sim = [return_dict["sim"][:, i] for i in range(target_horizons)]
            times = [return_dict["time"][:, i] for i in range(target_horizons)]
            return_dict.pop("obs")
            return_dict.pop("sim")
            return_dict.pop("time")

            #  update ONE per target horizon
            [
                return_dict.update({f"obs_{i + 1}": obs[i]})
                for i in range(target_horizons)
            ]
            [
                return_dict.update({f"sim_{i + 1}": sim[i]})
                for i in range(target_horizons)
            ]
            [
                # return_dict.update({f"time{'_' + str(i + 1) if i > 0 else ''}": times[i]})
                return_dict.update({f"time_{i + 1}": times[i]})
                for i in range(target_horizons)
            ]
        return return_dict

    def _get_horizons(self) -> List[int]:
        if self.cfg.horizon <= 1:
            return [self.cfg.horizon]
        else:
            return [i for i in range(1, self.cfg.horizon + 1)]

    def _test_epoch(self) -> DefaultDict[str, List[np.ndarray]]:
        out = defaultdict(list)
        with torch.no_grad():
            test_pbar = tqdm(self.test_dl, desc="Test set Forward Pass: ")
            for data in test_pbar:
                #  to GPU
                data = _to_device(data, self.device)

                x, y = data["x_d"], data["y"]
                y_hat = self.model(data)

                sim = y_hat["y_hat"].detach().cpu().numpy()
                obs = y.detach().cpu().numpy()

                #  -- Recreate the output data with metadata -- #
                pixels, times, horizons = create_metadata_arrays(data, self.test_dl)

                #  TODO: check that these reshapes work correctly
                sim = sim.reshape(pixels.shape)
                obs = obs.reshape(pixels.shape)

                out["horizon"].append(horizons)
                out["time"].append(times)
                out["pixel"].append(pixels)
                out["sim"].append(sim)
                out["obs"].append(obs)

        return out

    @staticmethod
    def test_default_dict_to_xarray(
        out: DefaultDict[str, List[np.ndarray]]
    ) -> xr.Dataset:
        return_dict: Dict[str, np.ndarray] = {}
        for key in out.keys():
            # concatenate over batch dimension (dimension = 0)
            var_ = np.concatenate(out[key])
            var_ = var_.squeeze() if var_.ndim == 3 else var_
            var_ = _reshape(var_)
            return_dict[key] = var_.flatten()

        preds = (
            pd.DataFrame(return_dict)
            .set_index(["time", "horizon", "pixel"])
            .to_xarray()
        )
        return preds

    def run_test(
        self,
        epoch: Optional[int] = None,
        unnormalize: bool = True,
        plot: bool = True,
        save_preds: bool = True,
    ) -> xr.Dataset:
        weight_file = self._get_weight_file(self.cfg, epoch=epoch)
        epoch = int(weight_file.name.split(".")[0][-3:])
        self.model.load_state_dict(
            torch.load(weight_file, map_location=self.cfg.device)
        )

        self.model.eval()
        out = self._test_epoch()
        preds = self.test_default_dict_to_xarray(out)

        # unnormalize values
        if unnormalize:
            normalizer = self.test_dl.dataset.normalizer
            preds = normalizer.unnormalize_preds(preds=preds, cfg=self.cfg)

        #  scatter plot the predictions
        if plot:
            for horizon in preds.horizon.values.reshape(-1):
                scatter_plot(preds, self.cfg, model="lstm", horizon=horizon)

        #  save the outputs
        if save_preds:
            preds.to_netcdf(
                self.cfg.run_dir
                / f"{self.subset}_predictions_E{str(epoch).zfill(3)}.nc"
            )

        return preds
