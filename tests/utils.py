import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from spatio_temporal.config import Config
from spatio_temporal.data.dataloader import PixelDataLoader


def _make_dataset(
    latlons: int = 10,
    lonmin: float = -180.0,
    lonmax: float = 180.0,
    latmin: float = -55.152,
    latmax: float = 75.024,
    start_date: str = "01-01-2000",
    end_date: str = "01-01-2001",
    variable_names: List[str] = ["feature", "target"],
):
    lat_len = lon_len = latlons
    longitudes = np.linspace(lonmin, lonmax, lon_len)
    latitudes = np.linspace(latmin, latmax, lat_len)
    times = pd.date_range(start_date, end_date, name="time", freq="M")
    dims = ["lat", "lon", "time"]
    coords = {"lat": latitudes, "lon": longitudes, "time": times}

    n_features = len(variable_names)
    data = np.random.randint(100, size=(latlons, latlons, len(times), n_features))

    ds = xr.Dataset(
        {
            var_name: (dims, data[:, :, :, ix])
            for ix, var_name in enumerate(variable_names)
        },
        coords=coords,
    )
    return ds


def get_oxford_weather_data() -> pd.DataFrame:
    """Oxford's weather which is hosted by Saad Jbabdi
    https://users.fmrib.ox.ac.uk/~saad/
    """
    names = [
        "year",
        "month",
        "max_temp",
        "min_temp",
        "frost_hrs",
        "precipitation",
        "sun_hrs",
    ]
    df = pd.read_csv(
        "https://users.fmrib.ox.ac.uk/~saad/ONBI/OxfordWeather.txt",
        delim_whitespace=True,
        header=None,
        names=names,
    )
    return df


def create_test_oxford_run_data(data_path: Path("data")):
    df = get_oxford_weather_data()
    df["time"] = pd.to_datetime(df["Date Time"])
    df["sample"] = 1
    df = df.drop("Date Time", axis=1)
    df = df.set_index(["time", "sample"])
    df.sort_index()
    df = df.sort_index()
    df = df.iloc[0:1000]
    df.to_csv(data_path / "test_oxford_weather.csv")


def download_test_jena_data(data_dir: Path = Path("data")) -> None:
    import tensorflow as tf
    import os

    zip_path = tf.keras.utils.get_file(
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        fname=(data_dir / "jena_climate_2009_2016.csv.zip").as_posix(),
        extract=True,
    )
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    df.to_csv(data_dir / "jena_climate_2009_2016.csv")


def get_jena_data() -> pd.DataFrame:
    df = pd.read_csv("data/test_jena_weather.csv")
    return df


def load_test_jena_data_as_dataset() -> xr.Dataset:
    df = get_jena_data()
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename({"sample": "pixel"}, axis=1).set_index(["time", "pixel"])
    ds = df.to_xarray()
    return ds


def create_and_assign_temp_run_path_to_config(cfg: Config, tmp_path: Path) -> None:
    # create run_dir
    (tmp_path / "runs").mkdir(exist_ok=True, parents=True)
    cfg.run_dir = tmp_path / "runs"


def create_sin_with_different_phases():
    assert False


def create_linear_ds(
    horizon: int = 1, alpha: float = 0, beta: float = 3.5, epsilon_sigma: float = 0
):
    ds = _make_dataset(start_date="01-01-2000", end_date="01-01-2021")

    def f(
        x: xr.DataArray, alpha: float, beta: float, epsilon_sigma: float
    ) -> xr.DataArray:
        epsilon = np.random.normal(loc=0, scale=epsilon_sigma, size=x.shape)
        y = alpha + (x * beta) + epsilon
        return y

    ds["target"] = f(ds["feature"], alpha=alpha, beta=beta, epsilon_sigma=epsilon_sigma)
    ds["feature"] = ds["feature"].shift(time=-horizon)

    if epsilon_sigma == 0:
        #  if no noise make sure that the calculation is correct
        yhat = alpha + (ds["feature"].shift(time=horizon) * beta)
        mask = yhat.isnull()
        yhat = yhat.values[~mask.values]
        y = ds["target"].values[~mask.values]
        assert np.all(yhat == y)

    return ds


def create_dummy_vci_ds():
    ds = _make_dataset(
        start_date="01-01-2000",
        end_date="01-01-2021",
        variable_names=["precip", "boku_VCI", "t2m", "SMsurf"],
    )

    def f(
        x: xr.Dataset, betas: np.ndarray = np.array([1, 2, 3]), epsilon_sigma: float = 1
    ) -> xr.DataArray:
        # y = X @ B + epsilon
        epsilon = np.random.normal(
            loc=0, scale=epsilon_sigma, size=tuple(dict(ds.dims).values())
        )
        x = x.to_array().values
        y = np.einsum("ijkl,i", x, betas) + epsilon
        dims = [k for k in dict(ds.dims).keys()]
        return (dims, y)

    ds["boku_VCI"] = f(ds[["precip", "t2m", "SMsurf"]])

    return ds
