from torch import Tensor
from typing import Dict, Tuple, Union, Any
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from spatio_temporal.data.dataloader import PixelDataLoader
from spatio_temporal.config import Config
from spatio_temporal.data.normaliser import Normalizer


def _create_dict_data_coords_for_individual_sample(
    y_hat: Tensor, y: Tensor
) -> Dict[str, Tuple[str, np.ndarray]]:
    # a tuple of (dims, data) ready for xr.Dataset creation
    #  TODO: forecast horizon ... ?
    data = {}
    y = y.view(1, -1).detach().numpy() if isinstance(y, Tensor) else y.reshape(1, -1)
    data["obs"] = (("pixel", "time"), y)
    y_hat = (
        y_hat.view(1, -1).detach().numpy()
        if isinstance(y_hat, Tensor)
        else y_hat.reshape(1, -1)
    )
    data["sim"] = (("pixel", "time"), y_hat)
    return data


def convert_individual_to_xarray(
    data: Dict[str, Union[Tensor, Any]],
    y_hat: Tensor,
    forecast_horizon: int,
    dataloader,
) -> xr.Dataset:
    # back convert to xarray object ...
    data_xr = _create_dict_data_coords_for_individual_sample(
        y_hat=y_hat["y_hat"], y=data["y"]
    )
    index = int(data["meta"]["index"])
    times = (
        data["meta"]["target_time"].detach().numpy().astype("datetime64[ns]").squeeze()
    )
    times = times.reshape(-1) if times.ndim == 0 else times
    pixel, _ = dataloader.dataset.lookup_table[int(index)]

    ds = xr.Dataset(data_xr, coords={"time": times, "pixel": [pixel]})
    return ds


def unnormalize_ds(
    dataloader: PixelDataLoader, ds: xr.Dataset, cfg: Config, normalizer: Normalizer
) -> xr.Dataset:
    pixel = str(ds.pixel.values[0])
    unnorm = normalizer.individual_inverse(
        ds, pixel_id=pixel, variable=cfg.target_variable
    )
    return unnorm


def get_individual_prediction_xarray_data(
    data: Dict[str, Union[Tensor, Any]],
    y_hat: Tensor,
    dataloader: PixelDataLoader,
    cfg: Config,
) -> xr.Dataset:
    ds = convert_individual_to_xarray(
        data=data, y_hat=y_hat, forecast_horizon=cfg.horizon, dataloader=dataloader
    )
    #  unnormalize the data (output scale)
    ds = unnormalize_ds(
        dataloader=dataloader, ds=ds, cfg=cfg, normalizer=dataloader.normalizer
    )
    #  correct the formatting
    if "sample" in ds.coords:
        ds = ds.drop("sample")
    return ds


def data_in_memory_to_xarray(
    data: Dict,
    y_hat: Union[np.ndarray, Tensor],
    cfg: Config,
    dataloader: PixelDataLoader,
) -> xr.Dataset:
    y = data["y"]
    y = y.detach().numpy() if isinstance(y, Tensor) else y
    y_hat = y_hat.detach().numpy() if isinstance(y_hat, Tensor) else y_hat

    times = (
        data["time"].detach().numpy()
        if isinstance(data["time"], Tensor)
        else data["time"]
    )
    times = pd.to_datetime(times.astype("datetime64[ns]").flatten())

    # TODO: generalise for other spatial dimensions too (e.g. basins)
    pixels = [
        dataloader.dataset.lookup_table[int(idx)][0] for idx in data["index"].flatten()
    ]
    lats = np.array([float(pixel.split("_")[0]) for pixel in pixels])
    lons = np.array([float(pixel.split("_")[1]) for pixel in pixels])

    #  TODO: deal with the cases where forecast horizon > 1 (and so multiple forecasts)
    if y.squeeze().ndim == 1:
        df = pd.DataFrame(
            dict(
                obs=y.squeeze(),
                sim=y_hat.squeeze(),
                lat=lats.squeeze(),
                lon=lons.squeeze(),
                time=times,
            )
        )
    else:
        # TODO: need to unpack multiple forecast horizons
        assert False
    ds = df.set_index(["time", "lat", "lon"]).to_xarray()

    #  TODO: unnormalize the raw xarray data
    normalizer = dataloader.dataset.normalizer
    ds = normalizer.transform_target_preds_Dataset(ds, cfg)
    return ds


def scatter_plot(preds: xr.Dataset, cfg: Config, model: str = "nn") -> None:
    f, ax = plt.subplots()
    ax.scatter(
        preds.obs.values.flatten(), preds.sim.values.flatten(), marker="x", alpha=0.1
    )
    ax.set_xlabel("Observations")
    ax.set_ylabel("Simulations")
    ax.set_title(f"{model} Observed vs. Predicted")

    f.savefig(cfg.run_dir / f"scatter_{model}.png")
