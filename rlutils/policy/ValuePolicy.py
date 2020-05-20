#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from abc import abstractmethod

from .Policy import Policy
from ..agent.Agent import Agent


class ValuePolicy(Policy):
    """
    Base class for all policies that select actions using an agent's Q-values. A finite number of actions is assumed.

    To implement a policy, sub-classes implement the method _select_action.
    """

    def __init__(self, agent: Agent):
        """

        :param agent: Agent this policy uses to select actions.
        :param eps: Epsilon parameter.
        """
        self._agent = agent

    @abstractmethod
    def _select_action(self, state, q_values):  # pragma: no cover
        """
        Method implemented by sub-classes that implement a specific policy.

        :param state: State at which action should be selected.
        :param q_values: Q-value array at state for each action.
        :return: Selected action.
        """
        raise NotImplementedError('select_action needs to be implemented by a subclass.')

    def __call__(self, state):
        """

        :param state: State for which action should be selected.
        :return: Action that is selected at the given state.
        """
        return self._select_action(state, self._agent.q_values(state))
