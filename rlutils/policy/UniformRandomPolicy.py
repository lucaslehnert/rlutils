#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np

from .ValuePolicy import ValuePolicy
from ..agent import ZeroValueAgent


class UniformRandomPolicy(ValuePolicy):
    """
    A policy which selects actions uniformly at random.
    """
    def __init__(self, agent):
        self._agent = agent

    def _select_action(self, state, q_values):
        return np.random.choice(range(len(q_values)))

def uniform_random(num_actions):
    """
    Returns a policy selecting actions uniformly at random.

    :param num_actions:
    :return:
    """
    return UniformRandomPolicy(ZeroValueAgent(num_actions))
