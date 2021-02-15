import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import List


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
