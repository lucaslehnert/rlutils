![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![pytest](https://github.com/lucaslehnert/rlutils/workflows/pytest/badge.svg)

# rlutils: A utility package for implementing reinforcement learning simulations.

rlutils is a python package containing utility functions to implement reinforcement learning simulations. 
The goal of rlutils is to provide a useful base library to help implement reinforcement learning (RL) research projects.
The `examples` directory contains jupyter notebooks demonstrating how to use rlutils.

rlutils provides an API to implement RL agents, policies, and simulate them in a control task (also called environment or Markov Decision Process (MDP)). rlutils also provides implementations to some basic policies, value iteration, and Q-learning. Other utility functions are also included to collect trajectories, store them, and replay them.
rlutils' goal is to provide a framework and utility functions for implementing RL algorithms.

If you have any questions about rlutils, please send email to `lucas_lehnert@brown.edu`.

## Installation

rlutils is tested on python 3.5, 3.6, 3.7, and 3.8.

This package can be installed by cloning this repository and then installing it using pip:

```
git clone https://github.com/lucaslehnert/rlutils.git
cd rlutils
pip install .
```
