import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from spatio_temporal.data.data_utils import encode_sample_str_as_int


if __name__ == "__main__":
    LEVEL = 3

    #  set up the base dir
    # base_dir = Path(".").absolute().parents[0]
    base_dir = Path("/home/tommy/spatio_temporal")
    assert base_dir.name == "spatio_temporal"

    if sys.path[0] != base_dir.as_posix():
        sys.path = [base_dir.as_posix()] + sys.path

    #  Load in the functions from the concept_formation package
    import sys

    assert (base_dir.parents[0] / "concept_formation").exists()
    sys.path.append(base_dir.parents[0] / "concept_formation")
    from scripts.data_process.clip_netcdf_to_shapefile import (
        prepare_rio_data,
        rasterize_all_geoms,
        create_timeseries_of_masked_datasets,
    )

    #  load in the data
    ds = xr.open_dataset(base_dir / "data/data_india_full.nc")
    gdf = gpd.read_file(base_dir / f"data/india/IND_adm{LEVEL}.shp")

    # ------------ CREATE THE TIMESERIES --------------
    #  rasterize the shapefile
    print("--- Rasterize the Geometries ---")
    ds, gdf = prepare_rio_data(ds, gdf)
    masks = rasterize_all_geoms(
        ds=ds,
        gdf=gdf,
        id_column=f"NAME_{LEVEL}",
        shape_dimension="region",
        geometry_column="geometry",
    )

    # create timeseries from mask geoms
    print("--- Mask the Geometries from the Data ---")
    out_ds = create_timeseries_of_masked_datasets(
        ds=ds, masks=masks, shape_dimension="region"
    )

    # encode the string from the "NAME_{LEVEL}" column
    encode_ds, lookup = encode_sample_str_as_int(out_ds, sample_str="region")

    print(f"--- Saving Data to {base_dir / 'data'} ---")
    pickle.dump(lookup, (base_dir / f"data/india_regionL{LEVEL}_lookup.pkl").open("wb"))
    encode_ds.to_netcdf(base_dir / f"data/data_india_regionsL{LEVEL}.nc")
