#  https://help.jasmin.ac.uk/article/4851-jasmin-notebook-service
#
import xarray as xr
import subprocess
from pathlib import Path
import numpy as np


station_ids = [
    24004,
    25006,
    27033,
    27073,
    27084,
    41022,
    43014,
    46005,
    47009,
    48003,
    48004,
    49004,
    50002,
    53009,
    53017,
    54018,
    72005,
    72014,
    73011,
    73015,
    8004,
    8009,
    12005,
    16003,
    20007,
    21017,
    21024,
    79002,
    83006,
    90003,
    94001,
    96002,
]


def load_level_data(data_dir: Path) -> xr.Dataset:
    list_ds = [xr.open_dataset(fp) for fp in (data_dir / "level_data").glob("*.nc")]
    ds = xr.concat(list_ds, dim="station")

    return ds


def copy_jasmin_level_data_to_home_dir(user_str: str = "chri4118"):
    base_dir = Path(f"/home/users/{user_str}")
    level_dir = base_dir / "level_data"
    level_dir.mkdir(exist_ok=True, parents=True)

    from_dir = Path(
        "/gws/nopw/j04/hydro_jules/data/uk/calval/river_flow_level/subdaily"
    )

    for station_id in station_ids:
        from_filepath = from_dir / f"{station_id:06}.nc"
        # print(station_id, from_filepath.exists())
        to_filepath = level_dir / f"{station_id:06}.nc"

        print(f"Running: `cp {from_filepath.as_posix()} {to_filepath.as_posix()}`")
        subprocess.run(["cp", from_filepath.as_posix(), to_filepath.as_posix()])


def scp_paramiko():
    # https://waterprogramming.wordpress.com/2021/03/17/automate-remote-tasks-with-paramiko/
    import paramiko

    # establish connection
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(
        hostname="remotehose", username="yourusername", password="yourpassword"
    )
    #  run commands
    stdin, stdout, stderr = ssh_client.exec_command("ls")


if __name__ == "__main__":
    data_dir = Path("data")

    if not (data_dir / "ALL_level_data.nc").exists():
        ds = load_level_data(data_dir)
        ds = ds.resample(time="D").mean()
        ds.to_netcdf(data_dir / "ALL_level_data.nc")
    else:
        ds = xr.open_dataset(data_dir / "ALL_level_data.nc")

    #  fix the coordinates
    station_id_dim = ds["station_id"].astype("int").values[0, :]
    ds["station"] = station_id_dim
    ds = (
        ds.drop("station_id")
        .rename({"station": "station_id"})
        .drop(["easting", "northing"])
    )

    # match with the camels data
    camels = camels.sel(
        station_id=np.isin(camels.station_id.values, ds.station_id.values)
    )
    ds = ds.sel(time=camels.time)

    camels["stage_value"] = ds["stage_value"]


"""notebook

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle 
base_dir = Path(".").absolute().parents[0]
import sys

if sys.path[0] != base_dir.as_posix():
    sys.path = [base_dir.as_posix()] + sys.path

from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.training.tester import Tester
from tests.utils import (
    create_linear_ds,
    _test_sklearn_model,
    download_test_jena_data,
    load_test_jena_data_as_dataset
)
from spatio_temporal.training.eval_utils import _plot_loss_curves, save_losses

ds = xr.open_dataset(base_dir / "data/camels_river_level_data.nc")
ds

RUN = [f.name for f in (base_dir / "runs").glob("river_level*")][0]

run_dir = base_dir / f"runs/{RUN}/"
cfg = Config(run_dir / "config.yml")
cfg.run_dir = run_dir

import torch

trainer = Trainer(cfg, ds)
model = trainer.model
model.load_state_dict(
    torch.load(sorted(list(run_dir.glob("*model*.pt")))[-1])
)

tester = Tester(cfg, ds, subset="test")
preds = tester.run_test(save_preds=False)

"""
