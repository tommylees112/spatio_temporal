#  TODO: spatial wide mean vs. region wide mean
from typing import Optional, List, Union
import xarray as xr
import torch
import numpy as np
import pandas as pd
from spatio_temporal.config import Config


class Normalizer:
    def __init__(
        self,
        fit_ds: Optional[xr.Dataset] = None,
        collapse_dims: Optional[List[str]] = ["time"],
    ):
        self.mean_: xr.Dataset
        self.std_: xr.Dataset
        self.collapse_dims = collapse_dims

        # if "sample" in fit_ds.data_vars:
        # fit_ds = fit_ds.rename({"sample": "pixel"})
        if fit_ds is not None:
            # 'train' mode
            self.fit(fit_ds, collapse_dims=collapse_dims)
        else:
            assert isinstance(self.mean_, xr.Dataset)
            assert isinstance(self.std_, xr.Dataset)

    @staticmethod
    def _update_xarray_value(ds: xr.Dataset, variable: str, value: float) -> xr.Dataset:
        shape = tuple([shape for shape in ds.dims.values()])
        key = tuple([key for key in ds.dims.keys()])
        values = (key, np.array(value).repeat(shape).astype(float))

        ds[variable] = values
        return ds

    def update_mean_with_constant(self, variable: str, mean_value: float) -> None:
        updated_mean = self.mean_.copy()
        updated_mean = self._update_xarray_value(updated_mean, variable, mean_value)
        self.mean_ = updated_mean

    def update_std_with_constant(self, variable: str, std_value: float) -> None:
        updated_std = self.std_.copy()
        updated_std = self._update_xarray_value(updated_std, variable, std_value)
        self.std_ = updated_std

    def fit(self, fit_ds: xr.Dataset, collapse_dims: Optional[List[str]] = ["time"]):
        self.mean_ = fit_ds.mean(dim=collapse_dims)
        self.std_ = fit_ds.std(dim=collapse_dims)
        self._check_std()
        print("** Normalizer fit! **")

    def _check_std(self, epsilon: float = 1e-10):
        #  TODO: keep constant values as their raw value ?
        #  replace std values close to zero with small number (epsilon)
        std_zero = xr.ones_like(self.std_.to_array()) * np.isclose(
            self.std_.to_array().values, 0
        )
        std_zero = std_zero.to_dataset(dim="variable").astype("bool")
        self.std_ = self.std_.where(~std_zero, epsilon)

    def transform(self, ds: xr.Dataset, variables: Optional[List[str]]) -> xr.Dataset:
        # zero mean, unit variance
        if variables is not None:
            norm_list = []
            for var in variables:
                norm_list = (ds[var] - self.mean_[var]) / self.std_[var]
            normed = xr.merge(norm_list)
        else:
            #  normalize ALL variables
            normed = (ds - self.mean_) / self.std_

        return normed

    ########################################################
    ########### RECOVERING UNDERLYING DATA #################
    ########################################################
    def inverse_transform(
        self, ds: xr.Dataset, variables: Optional[List[str]]
    ) -> xr.Dataset:
        if variables is not None:
            norm_list = []
            for var in variables:
                norm_list = (ds[var] * self.std_[var]) + self.mean_[var]
            unnormed = xr.merge(norm_list)
        else:
            #  unnormalize ALL variables
            unnormed = (ds * self.std_) + self.mean_

        return unnormed

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

        #  only ever one forecast_horizon as output
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
