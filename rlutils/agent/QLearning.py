#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
from typing import Optional, Dict, Any
from .Agent import Agent


class QLearning(Agent):
    """
    Implementation of Q-learning, as described by Watkins and Dayan 1992.
    """

    def __init__(
        self, 
        state_column_name: str, 
        num_states: int, 
        num_actions: int, 
        learning_rate: float, 
        gamma: float=0.9, 
        init_Q: Optional[np.ndarray]=None
    ):
        """Constructs a Q-learning agent.

        Args:
            state_column_name (str): State column name containing state index.
            num_states (int): Number of states.
            num_actions (int): Number of Actions.
            learning_rate (float): Learning rate.
            gamma (float, optional): Discount factor. Defaults to 0.9.
            init_Q (Optional[np.ndarray], optional): Initial Q-values. Defaults 
                to None.
        """
        self._state_column_name = state_column_name
        self._num_states = num_states
        self._num_actions = num_actions
        self._lr = learning_rate
        self._gamma = gamma
        if init_Q is None:
            self._init_Q = np.ones([self._num_actions, self._num_states]) 
            self._init_Q *= 1. / (1. - gamma)
        elif type(init_Q) is np.ndarray:
            assert (np.shape(init_Q) == (self._num_actions, self._num_states))
            self._init_Q = init_Q
        else:
            self._init_Q = np.ones([self._num_actions, self._num_states]) 
            self._init_Q *= init_Q

        self._q = np.array(self._init_Q, copy=True)

    def reset(self, q_vec: Optional[np.ndarray]=None):
        """Reset Q-learning agent.

        Args:
            q_vec (Optional[np.ndarray], optional): Q-value table. Defaults to 
                None.
        """
        if q_vec is None:
            self._q = np.array(self._init_Q, copy=True)
        else:
            self._q = np.array(q_vec, copy=True)

    def q_values(self, state: Dict[str, Any]):
        return self._q[:, state[self._state_column_name]]

    def update_transition(
        self, 
        state: Dict[str, Any], 
        action: int, 
        reward: float, 
        next_state: Dict[str, Any], 
        term: bool, 
        info: Dict[Any, Any]
    ): # pragma: no cover
        state_idx = state[self._state_column_name]
        next_q_vals = self._q[:, next_state[self._state_column_name]]
        target = reward + (1. - term) * self._gamma * np.max(next_q_vals)
        td_error = target - self._q[:, state_idx]
        self._q[action, state_idx] += self._lr * td_error
        return {'td_error': td_error}

    def on_simulation_timeout(self):
        pass

    def get_q_vector(self) -> np.ndarray:
        """

        :return: Array of shape [num_actions, num_states] containing all Q-values.
        """
        return np.copy(self._q)

    def get_gamma(self) -> float:
        """Discount factor getter

        :return: Discount factor gamma.
        :rtype: float
        """
        return self._gamma

    def get_learning_rate(self) -> float:
        """

        :return: Learning rate.
        """
        return self._lr
