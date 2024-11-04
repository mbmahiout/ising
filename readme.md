

# Project Ising

This is the codebase accompanying my masters thesis at the Kavli Institute for Systems Neuroscience at NTNU, titled **A Statistical Physics Approach to
Modelling the Joint Activity of Cortical Populations**.

It was originally in Python, but was re-written with the core functionality in C++ for greater performance. It includes implementations of the equilibrium and non-equilibrium Ising models, as well as methods for generating MCMC simulations and inference methods used to fit the models to neural recording data. I've used C++ for the "back-end", which includes all of the aforementioned core functionality. Additionally, there is a python "front-end" for data processing, visualization, and wrappers for some of the C++ code. Finally, pybind11 was used to form a bridge from the C++ component to the Python one.

## Installation

### Prerequisites
- C++20
- CMake >= 3.27
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [PyBind11](https://github.com/pybind/pybind11)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)

### Setting up Python environment
```
conda create -n ising_env python=3.12.1
conva activate ising_env
pip install -r requirements.txt
```

### Building the C++ executable
```
cd build
cmake ..
make
cd ..
```

## Structure:
- C++ code for MCMC simulations, gradient ascent to maximize likelihood, as well as pybindings for making the C++ modules avaliable in Python is found in the ./src directory, and the associated headers are situated in ./include.
- The Python code, which includes classes and functions for data processing, visualization, and wrappers for the modules and classes is found in ./python/src. Here, you'll also find a collection of notebooks that were used for data visualization and exploratory analysis.
- Having installed the requirements and proceeded with the set-up described above, the notebooks eq_inverse_testing.ipynb and neq_inverse_testing.ipynb (found under simulation_testing) may be a good place to start exploring the models.

## Data
I'm not at liberty to make the calcium imaging recordings avaliable, however, the data directory includes parameters fitted to the neural data for equilibrium and non-equilibrium systems ranging in size from 10 to 270 neurons.
