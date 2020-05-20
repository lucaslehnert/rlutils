#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np

from .ValuePolicy import ValuePolicy
from ..agent.Agent import Agent


class EGreedyPolicy(ValuePolicy):
    """
    Implementation of an epsilon-greedy policy. The policy selects actions uniformly at random with epsilon probability.
    With probability 1 - epsilon, actions are selected greedily.
    """

    def __init__(self, agent: Agent, eps):
        """

        :param agent: Agent this policy uses to select actions.
        :param eps: Epsilon parameter.
        """
        self._agent = agent
        self.set_epsilon(eps)

    def set_epsilon(self, eps):
        """

        :param eps: Epsilon parameter that lies in [0, 1].
        :return:
        """
        assert 0. <= eps <= 1.
        self._eps = eps

    def _select_action(self, state, q_values):
        opt_act = np.where(np.max(q_values) == q_values)[0]
        if np.random.choice([False, True], p=[1. - self._eps, self._eps]):
            return np.random.choice(range(len(q_values)))
        else:
            return np.random.choice(opt_act)
