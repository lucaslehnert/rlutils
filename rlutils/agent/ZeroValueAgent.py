#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np

from .ValueFunctionAgent import ValueFunctionAgent


class ZeroValueAgent(ValueFunctionAgent):
    """
    Implementation of an agent the uses a zero value function.
    """

    def __init__(self, num_actions):
        """
        An agent that provides zero Q-values. This agent could be used for uniform random action selection.

        :param num_actions: Number of actions to select from.
        """
        super().__init__(q_fun=lambda _: np.zeros(num_actions))


UniformActionSelectionAgent = ZeroValueAgent
