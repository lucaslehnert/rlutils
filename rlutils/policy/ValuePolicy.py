#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
from abc import abstractmethod
from ..types import Policy, Agent
from typing import Dict, Any



class ValuePolicy(Policy):
    """
    Base class for all policies that select actions using an agent's Q-values. A
    finite number of actions is assumed.

    To implement a policy, sub-classes implement the method _select_action.
    """

    def __init__(self, agent: Agent):
        """

        :param agent: Agent this policy uses to select actions.
        :param eps: Epsilon parameter.
        """
        self._agent = agent

    @abstractmethod
    def _select_action(
        self, 
        state: Dict[str, Any], 
        q_values: np.ndarray
    ) -> Any:  # pragma: no cover
        """Method implemented by sub-classes that implement a specific policy.

        Args:
            state (Dict[str, Any]): _description_
            q_values (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Any: _description_
        """
        raise NotImplementedError(
            'select_action needs to be implemented by a subclass.')

    def __call__(self, state: Dict[str, Any]) -> Any:
        return self._select_action(state, self._agent.q_values(state))
