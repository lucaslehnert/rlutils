#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from abc import abstractmethod

from rlutils.data.simulate import TransitionListener


class Agent(TransitionListener):
    """
    Super class for all value-based agents.
    """

    @abstractmethod
    def q_values(self, state):  # pragma: no cover
        """
        Q-values for a given state.

        :param state: State of an MDP.
        :return: A vector of dimension [num_actions]. Each entry contains the Q-values for each action at the provided
            state.
        """

    @abstractmethod
    def reset(self, *params, **kwargs):  # pragma: no cover
        """
        Reset agent to its initialization.
        :return:
        """
