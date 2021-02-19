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
    #  TODO: how to deal with gpu in a more realistic way
    data = {}
    y = y.view(1, -1).detach().cpu().numpy()
    y = y if isinstance(y, Tensor) else y.reshape(1, -1)
    data["obs"] = (("pixel", "time"), y)
    y_hat = (
        y_hat.view(1, -1).detach().cpu().numpy()
        if isinstance(y_hat, Tensor)
        else y_hat.reshape(1, -1)
    )
    data["sim"] = (("pixel", "time"), y_hat)
    return data


def get_lists_of_metadata(
    data: Dict[str, Union[Tensor, Any]], dataloader
) -> Tuple[np.ndarray, ...]:
    indexes = [int(ix) for ix in data["meta"]["index"]]
    times = (
        data["meta"]["target_time"]
        .detach()
        .cpu()
        .numpy()
        .astype("datetime64[ns]")
        .squeeze()
    )
    times = times.reshape(-1) if times.ndim == 0 else times

    pixels = np.array(
        [dataloader.dataset.lookup_table[int(index)][0] for index in indexes]
    )
    # TODO: fix this hack (maybe remove times from being stored in data)
    if times.size == 0:
        target_ixs = [int(dataloader.dataset.lookup_table[int(index)][-1]) for index in indexes]
        times_ = []
        for target_ix, pixel in zip(target_ixs, pixels):
            times_.append(dataloader.dataset.times[pixel][target_ix])
        times = np.array(times_).astype("datetime64[ns]")

    return pixels, times


def convert_individual_to_xarray(
    data: Dict[str, Union[Tensor, Any]],
    y_hat: Tensor,
    forecast_horizon: int,
    dataloader,
) -> xr.Dataset:
    # back convert to xarray object ...
    assert dataloader.batch_size < 2, "This method does not work for batch sizes > 1"
    times, pixels = get_lists_of_metadata(data, dataloader)

    # reshape data to N PIXELS; N TIMES
    n_pixels = len(np.unique(pixels))
    n_times = len(np.unique(times))

    data_xr = _create_dict_data_coords_for_individual_sample(
        y_hat=y_hat["y_hat"].reshape(n_pixels, n_times),
        y=data["y"].reshape(n_pixels, n_times),
    )

    ds = xr.Dataset(data_xr, coords={"time": times, "pixel": np.unique(pixels)})
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
    y = y.detach().cpu().numpy() if isinstance(y, Tensor) else y
    y_hat = y_hat.detach().cpu().numpy() if isinstance(y_hat, Tensor) else y_hat

    times = (
        data["time"].detach().cpu().numpy()
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
