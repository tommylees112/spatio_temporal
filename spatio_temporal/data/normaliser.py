from typing import Optional, List, Union
import xarray as xr
import torch
import numpy as np


class Normalizer:
    def __init__(
        self, fit_ds: Optional[xr.Dataset] = None, collapse_dims: List[str] = ["time"]
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

    def transform_original_Dataset(ds: xr.Dataset):
        # TODO: convert pixel ("{lat}_{lon}") to lat, lon information
        #  then convert as normal
        pass
