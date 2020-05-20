#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np

import rlutils.algorithm.vi as vi
from .ValueFunctionAgent import ValueFunctionAgent


class VIAgent(ValueFunctionAgent):
    """
    VIAgent performs value iteration on a tabular MDP when the object is constructed. These Q-values are then
    subsequently used by the agent subsequently. The reset and update_transition methods have no effect. The q_values
    method returns Q-values that were computed using value iteration.

    States have to be represented as one-hot bit vectors.
    """

    def __init__(self, t_mat, r_vec, gamma, eps=1e-5, max_it=5000):
        """
        An agent that uses a full transition matrix and reward vector to run value iteration and then use the computed
        value function for decision making.

        :param t_mat: A transition table of the format [num_actions, num_states, num_states]
        :param r_vec: An expected reward table of the format [num_actions, num_states].
        :param gamma: The used discount factor.
        :param eps: Epsilon used to detect termination in VI.
        :param max_it: Maximum iterations VI is run for. If threshold is reached, a VITimeoutException is thrown.
        """
        self._q, _ = vi(t_mat, r_vec, gamma, eps=eps, max_it=max_it)
        super().__init__(q_fun=lambda s: np.matmul(self._q, s))

