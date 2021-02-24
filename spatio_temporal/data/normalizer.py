from typing import Optional, List, Union
import xarray as xr
import torch
import numpy as np
import pandas as pd
from spatio_temporal.config import Config


class Normalizer:
    def __init__(
        self, fit_ds: Optional[xr.Dataset] = None, collapse_dims: List[str] = ["time"],
    ):
        # if "sample" in fit_ds.data_vars:
        # fit_ds = fit_ds.rename({"sample": "pixel"})
        if fit_ds is not None:
            # 'train' mode
            self.fit(fit_ds, collapse_dims=collapse_dims)
        else:
            assert isinstance(self.mean_, xr.Dataset)
            assert isinstance(self.std_, xr.Dataset)

    def fit(self, fit_ds: xr.Dataset, collapse_dims: List[str] = ["time"]):
        self.mean_ = fit_ds.mean(dim=collapse_dims)
        self.std_ = fit_ds.std(dim=collapse_dims)
        self._check_std()

    def _check_std(self, epsilon: float = 1e-10):
        #  replace std values close to zero with small number (epsilon)
        std_zero = xr.ones_like(self.std_.to_array()) * np.isclose(
            self.std_.to_array().values, 0
        )
        std_zero = std_zero.to_dataset(dim="variable").astype("bool")
        self.std_.where(~std_zero, epsilon)

    def transform(self, ds) -> xr.Dataset:
        # zero mean, unit variance
        return (ds - self.mean_) / self.std_

    def inverse_transform(self, ds: xr.Dataset) -> xr.Dataset:
        return (ds * self.std_) + self.mean_

    def individual_inverse(
        self, data: Union[np.ndarray, torch.Tensor], pixel_id: str, variable: str
    ):
        """inverse transform for one pixel and one variable"""
        std_ = self.std_.sel(sample=pixel_id)[variable].values
        mean_ = self.mean_.sel(sample=pixel_id)[variable].values

        return (data * std_) + mean_

    def unnormalize_preds(self, preds: xr.Dataset, cfg: Config) -> xr.Dataset:
        """Unnormalize the output of the Tester.test_run() process

        Args:
            preds (xr.Dataset): Created dataset from the Tester.test_run()
            cfg (Config): configuration for the current experiment

        Returns:
            xr.Dataset: Unnormalized obs/sim for the target variable
        """
        mean = self.mean_.rename({"sample": "pixel"})
        std = self.std_.rename({"sample": "pixel"})
        target = cfg.target_variable 

        # only ever one forecast_horizon as output
        unnorm_ds = preds.copy().isel(horizon=0)
        
        for variable in preds.data_vars:
            unnorm_ds[variable] = (unnorm_ds[variable] * std[target]) + mean[target]
        
        return unnorm_ds

    @staticmethod
    def _unpack_spatial_coords(ds: xr.Dataset, spatial_dims: List[str]) -> pd.DataFrame:
        df = ds.to_dataframe().reset_index()
        spatial_info = pd.DataFrame(
            dict(
                zip(
                    spatial_dims,
                    np.array([s.split("_") for s in df["sample"].to_list()]).T,
                )
            )
        )
        df = df.join(spatial_info).drop("sample", axis=1)
        return df

    def transform_target_preds_Dataset(self, ds: xr.Dataset, cfg: Config) -> xr.Dataset:
        # TODO: convert pixel ("{lat}_{lon}") to lat, lon information
        #  then convert as normal
        target = cfg.target_variable
        spatial_dims = cfg.pixel_dims
        unnorm_ds = ds.copy()

        std = (
            self._unpack_spatial_coords(self.std_, spatial_dims)
            .set_index(spatial_dims)
            .to_xarray()
        )
        mean = (
            self._unpack_spatial_coords(self.mean_, spatial_dims)
            .set_index(spatial_dims)
            .to_xarray()
        )

        #  Match the types of the spatial_dims (lat, lon)
        try:
            for dim in spatial_dims:
                std[dim] = std[dim].astype("float")
                mean[dim] = mean[dim].astype("float")
        except:
            #  ignore if it's a string that can't be converted e.g. region name
            pass

        # normalize each variable
        for variable in ds.data_vars:
            unnorm_ds[variable] = (unnorm_ds[variable] * std[target]) + mean[target]

        # unnorm_ds = (unnorm_ds * std[target]) + mean[target]
        return unnorm_ds
