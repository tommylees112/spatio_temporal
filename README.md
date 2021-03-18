## See: [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) 
# Python Library for training Neural Networks for spatio-temporal raster data pixel-wise

**Note**: Most of the code has been borrowed from the amazing team at [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) 

Indeed, the only innovation here has been the automatic creation of 2D data (`dims=["time", "pixel"]`) from 3D input data (`dims=["time", "lat", "lon"]`). 

Two methods for making this more maintainable in the future are:
1) Write a dataloader class as a Pull Request to `neuralhydrology`
1) Create a standalone package that inherits most behaviour from `neuralhydrology` but with differences as explained here

## Creating the environment to work with the code
NOTE: Requires [`miniconda`](https://docs.conda.io/en/latest/miniconda.html)
```bash
conda create -n ml --yes python=3.7
conda activate ml
```

```bash
conda install tensorflow -c anaconda --yes
conda install pytorch torchvision -c pytorch --yes
conda install -c conda-forge seaborn=0.11 --yes
conda install -c conda-forge netcdf4 numba tqdm jupyterlab tensorboard ipython pip ruamel.yaml xarray descartes statsmodels scikit-learn black mypy --yes
# pip install geopandas
```

## TODO: make the run.py more general
- read in dataset
- `Trainer` and `Tester` classes
```bash
ipython --pdb run.py train  -- --config_file tests/testconfigs/run_test_config.yml
```

## Run the tests to get a feel for how the pipeline works
```bash
pytest --pdb .
```

## TODO: test on various other 
- [pmdarima example timeseries datasets](http://alkaline-ml.com/pmdarima/modules/datasets.html)
```python
from pmdarima.datasets import load_airpassengers, load_austres, load_heartrate, load_lynx, load_taylor, load_wineind, load_woolyrnq, load_msft
```