from torch import Tensor
from typing import Dict, Tuple, Union, Any
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from spatio_temporal.data.dataloader import PixelDataLoader
from spatio_temporal.config import Config
from spatio_temporal.data.normalizer import Normalizer


def create_metadata_arrays(
    data: Dict[str, Dict[str, Union[Tensor, Any]]], dataloader: PixelDataLoader
) -> Tuple[np.ndarray, ...]:
    """Create metadata arrays for time, pixel_id 

    Args:
        data (Dict[str, Union[Tensor, Any]]): The batched data output by dataloader
        dataloader ([PixelDataLoader]): Dataloader which data read from 

    Returns:
        Tuple[np.ndarray, ...]: [description]
    """
    indexes = [int(ix) for ix in data["meta"]["index"]]
    times = (
        data["meta"]["target_time"]
        .detach()
        .cpu()
        .numpy()
        .astype("datetime64[ns]")
        .squeeze()
    )
    #  create (batch_size, n_target_times)
    horizon = dataloader.dataset.cfg.horizon
    n_target_times = 1
    #  TODO: len(times) offers up error?
    try:
        times = times.reshape(-1, n_target_times) if len(times) > 1 else times
    except TypeError as E:
        #  len() of unsized objet
        #    In[]:  times.shape
        #    Out[]: ()
        times = times.reshape(-1, n_target_times)

    # copy pixel arrays for each n_target_times
    pixels = np.array(
        [dataloader.dataset.lookup_table[int(index)][0] for index in indexes]
    )
    pixels = np.tile(pixels.reshape(-1, 1), n_target_times)

    #  get forecast horizons as another array
    forecast_horizon = horizon
    forecast_horizons = np.tile(forecast_horizon, pixels.shape[0]).reshape(pixels.shape)

    #  TODO: fix this hack (maybe remove times from being stored in data)
    if times.size == 0:
        target_ixs = [
            int(dataloader.dataset.lookup_table[int(index)][-1]) for index in indexes
        ]
        times_ = []
        for target_ix, pixel in zip(target_ixs, pixels):
            times_.append(dataloader.dataset.times[pixel][target_ix])
        times = np.array(times_).astype("datetime64[ns]")

    return pixels, times, forecast_horizons


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


def plot_1_1_line(x: np.ndarray, ax) -> plt.Axes:
    # plot 1:1 line
    line_1_1_x = np.linspace(x.min(), x.max(), 10)
    ax.plot(line_1_1_x, line_1_1_x, "k--", label="1:1 Line", alpha=0.5)
    return ax


def _plot_scatter(preds: xr.Dataset) -> Tuple[Any, Any]:
    f, ax = plt.subplots()
    ax.scatter(
        preds.obs.values.flatten(), preds.sim.values.flatten(), marker="x", alpha=0.1
    )
    ax.set_xlabel("Observations")
    ax.set_ylabel("Simulations")
    # set range of values to the same for both axis (obs and sim)
    axis = np.concatenate([np.array(ax.get_xlim()), np.array(ax.get_ylim())])
    ax.set_xlim(axis.min(), axis.max())
    ax.set_ylim(axis.min(), axis.max())
    ax = plot_1_1_line(axis, ax)
    return f, ax


def scatter_plot(
    preds: xr.Dataset, cfg: Config, model: str = "nn", horizon: int = 0
) -> None:
    preds = preds.drop("horizon")
    if "spatial_ref" in [c for c in preds.coords]:
        preds = preds.drop("spatial_ref")

    f, ax = _plot_scatter(preds)
    ax.set_title(f"{model} Observed vs. Predicted [FH {horizon}]")

    f.savefig(cfg.run_dir / f"scatter_{model}_FH{horizon}.png")


def _plot_loss_curves(losses: Tuple[np.ndarray, np.ndarray]) -> Tuple[Any, Any]:
    train_losses, valid_losses = losses
    f, ax = plt.subplots()
    ax.plot(train_losses, label="Train", color="C0", marker="x")
    ax.plot(valid_losses, label="Validation", color="C1", marker="x")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.legend()
    return f, ax


def _plot_single_timeseries(preds: xr.Dataset) -> Tuple[Any, Any]:
    f, ax = plt.subplots(figsize=(12, 4))
    pixel = np.random.choice(preds.pixel.values)
    if "horizon" in [c for c in preds.coords]:
        preds = preds.drop("horizon")
    if "spatial_ref" in [c for c in preds.coords]:
        preds = preds.drop("spatial_ref")
    preds.sel(pixel=pixel).to_dataframe().plot(ax=ax)
    plt.legend()
    ax.set_title(pixel)
    return f, ax


def save_timeseries(preds: xr.Dataset, cfg: Config, n: int = 1) -> None:
    for _ in range(n):
        f, ax = _plot_single_timeseries(preds)
        pixel = str(ax.get_title())
        f.savefig(cfg.run_dir / f"{pixel}_timeseries.png")


def save_loss_curves(losses: Tuple[np.ndarray, np.ndarray], cfg: Config) -> None:
    f, ax = _plot_loss_curves(losses=losses)
    f.savefig(cfg.run_dir / "loss_curves.png")


def save_losses(losses: Tuple[np.ndarray, np.ndarray], cfg: Config) -> None:
    train_losses, valid_losses = losses
    df = pd.DataFrame(
        {"train": train_losses.flatten(), "validation": valid_losses.flatten()}
    )
    df.to_csv(cfg.run_dir / "losses.csv")


def _fix_output_timestamps_monthly(preds: xr.Dataset) -> xr.Dataset:
    #  beacause of imprecise storage of datetime -> float
    preds["time"] = [pd.to_datetime(t).round("D") for t in preds.time.values]
    return preds
