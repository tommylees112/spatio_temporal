# Python Library for training LSTMs and other Neural Networks for spatio-temporal raster data

**Note**: Most of the code has been borrowed from the amazing team at [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) 

Indeed, the only innovation here has been the automatic creation of 2D data (`dims=["time", "pixel"]`) from 3D input data (`dims=["time", "lat", "lon"]`). 

Two methods for making this more maintainable in the future are:
1) Write a dataloader class as a Pull Request to `neuralhydrology`
1) Create a standalone package that inherits most behaviour from `neuralhydrology` but with differences as explained here

## Creating the environment to work with the package
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

## Models are not learning :(
Trying to learn a linear relationship between `features` and `target` from data created by `from tests.utils import create_linear_ds`.
```bash
ipython --pdb run.py train  -- --config_file tests/testconfigs/run_test_config.yml
```

Current output :sad:
```
Validation Epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████| 3300/3300 [00:00<00:00, 7339.99it/s]
Train Loss: 117.65
Valid Loss: 29.41
Training Epoch 2: 100%|████████████████████████████████████████████████████████████████████████| 15300/15300 [00:05<00:00, 2599.40it/s, 23817.25]
Validation Epoch 2: 100%|██████████████████████████████████████████████████████████████████████████████████| 3300/3300 [00:00<00:00, 7375.67it/s]
Train Loss: 112.18
Valid Loss: 54.74
Training Epoch 3: 100%|████████████████████████████████████████████████████████████████████████| 15300/15300 [00:05<00:00, 2599.48it/s, 23913.97]
Validation Epoch 3: 100%|██████████████████████████████████████████████████████████████████████████████████| 3300/3300 [00:00<00:00, 7327.49it/s]
Train Loss: 108.55
Valid Loss: 76.24
Training Epoch 4: 100%|████████████████████████████████████████████████████████████████████████| 15300/15300 [00:05<00:00, 2596.00it/s, 23996.17]
Validation Epoch 4: 100%|██████████████████████████████████████████████████████████████████████████████████| 3300/3300 [00:00<00:00, 7408.75it/s]
Train Loss: 105.99
Valid Loss: 94.19
Training Epoch 5: 100%|████████████████████████████████████████████████████████████████████████| 15300/15300 [00:05<00:00, 2601.24it/s, 24064.47]
Validation Epoch 5: 100%|██████████████████████████████████████████████████████████████████████████████████| 3300/3300 [00:00<00:00, 7394.89it/s]
Train Loss: 104.22
Valid Loss: 109.00
Test set Forward Pass: 100%|████████████████████████████████████████████████████████████████████████████████| 5700/5700 [00:07<00:00, 784.24it/s]
... Merging all predictions to one xr.Dataset
```