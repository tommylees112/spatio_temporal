Python Library for training LSTMs and other Neural Networks for pixel data

**Note**: Most of the code has been borrowed from the amazing team at [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) 

Indeed, the only innovation here has been the automatic creation of 2D data (`dims=["time", "pixel"]`) from 3D input data (`dims=["time", "lat", "lon"]`). 

Two methods for making this more maintainable in the future are:
1) Write a dataloader class as a Pull Request to `neuralhydrology`
1) Create a standalone package that inherits most behaviour from `neuralhydrology` but with differences as explained here

