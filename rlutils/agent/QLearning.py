#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
from .Agent import Agent


class QLearning(Agent):
    """
    Implementation of Q-learning, as described by Watkins and Dayan 1992.
    """

    def __init__(self, num_states, num_actions, learning_rate, gamma=0.9, init_Q=None):
        """
        Q-learning agent.

        :param num_states: Number of states. States are assumed to be represented as one-hot vectors.
        :param num_actions: Number of actions.
        :param learning_rate: Learning rate.
        :param gamma: Discount factor. Default is 0.9.
        :param init_Q: Q-value initialization. Default is None.
            If init_Q is None, then Q-values are initialized to 1/(1 - gamma). If init_Q is a scalar, then all Q-values
            are initialized to this scalar value. If init_Q is a np.ndarray, then it has to be of shape [num_actions,
            num_states]. This array is then used to reset all Q-values.
        """
        self._num_states = num_states
        self._num_actions = num_actions
        self._lr = learning_rate
        self._gamma = gamma
        if init_Q is None:
            self._init_Q = np.ones([self._num_actions, self._num_states]) * 1. / (1. - gamma)
        elif type(init_Q) is np.ndarray:
            assert (np.shape(init_Q) == (self._num_actions, self._num_states))
            self._init_Q = init_Q
        else:
            self._init_Q = np.ones([self._num_actions, self._num_states]) * init_Q

        self._q = np.array(self._init_Q, copy=True)

    def reset(self, q_vec=None):
        """
        Reset Q-learning agent.

        :param q_vec: Q-values used to overwrite learned Q-values. Default is None, in which case the initialization
            provided to the constructor is used.
        :return:
        """
        if q_vec is None:
            self._q = np.array(self._init_Q, copy=True)
        else:
            self._q = np.array(q_vec, copy=True)

    def q_values(self, state):
        return np.matmul(self._q, state)

    def update_transition(self, state, action, reward, next_state, term, info):
        """
        Update agent with a single step transition.

        :param state: State represented as a one-hot bit vector.
        :param action: Action represented as a zero-based index.
        :param reward: Reward given for a particular transition.
        :param next_state: Next state represented as a one-hot bit vector.
        :param term: Termination flag. This flag is ignored.
        :param info: Other transition info. This dictionary is ignored.
        :return: A dictionary {'td_error': td_error}, where td_error is the temporal difference error induced by the
            given transition.
        """
        target = reward + (1. - term) * self._gamma * np.max(np.matmul(self._q, next_state))
        td_error = target - np.dot(self._q[action], state)
        self._q[action] = self._q[action] + self._lr * td_error * state
        return {'td_error': td_error}

    def get_q_vector(self):
        """

        :return: Array of shape [num_actions, num_states] containing all Q-values.
        """
        return np.array(self._q)

    def get_gamma(self):
        """

        :return: Discount factor gamma.
        """
        return self._gamma

    def get_learning_rate(self):
        """

        :return: Learning rate.
        """
        return self._lr
