![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![pytest](https://github.com/lucaslehnert/rlutils/workflows/pytest/badge.svg)

# rlutils: A utility package for implementing reinforcement learning simulations

rlutils is a python package containing utility functions to implement 
reinforcement learning simulations.  The goal of rlutils is to provide a useful 
base library to help implement reinforcement learning (RL) research projects.
The `examples` directory contains jupyter notebooks demonstrating how to use 
rlutils.

rlutils provides an API to implement RL agents, policies, and simulate them in a
control task (also called environment or Markov Decision Process (MDP)). rlutils
also provides implementations to some basic policies, value iteration, and 
Q-learning. Other utility functions are also included to collect trajectories, 
store them, and replay them.

rlutils' goal is to provide a framework and utility functions for implementing 
RL algorithms.

If you have any questions about rlutils, please send email to 
`lucas_lehnert@brown.edu`.

## Installation

rlutils is tested on python 3.5, 3.6, 3.7, and 3.8.

This package can be installed using pip:

```
pip install rlutils
```

Alternatively, the repository can also be cloned and the package can be 
installed without pulling from PyPI:

```
git clone https://github.com/lucaslehnert/rlutils.git
cd rlutils
pip install .
```

## Generating API Documentation

The API documentation can be generated using Sphinx. To build the HTML 
documentation on Ubuntu 20.04, first install the following packages using pip:

```
sudo apt install pandoc
pip install Sphinx==3.5.2 sphinx-rtd-theme==0.5.1 nbsphinx pandoc
```

To update the API docs with sphinx run the following:

```
cd docs
sphinx-apidoc -o source .. ../setup.py ../test
```

The documentation can be build in `docs/build` by running the following:

```
cd docs/
make html
```

The resulting HTML documentation can be viewed by pointing your browser to open
the file `docs/build/index.html`.
