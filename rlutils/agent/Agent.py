#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
from abc import abstractmethod
from typing import Dict, Any
from ..data import TransitionListener, Column


class Agent(TransitionListener):
    """
    Super class for all value-based agents.
    """

    @abstractmethod
    def q_values(self, state: Dict[str, Any]) -> np.ndarray:  # pragma: no cover
        """
        Q-values for a given state.

        :param state: State of an MDP.
        :return: A vector of dimension [num_actions]. Each entry contains the 
            Q-values for each action at the provided state.
        """

    @abstractmethod
    def reset(self, *params, **kwargs):  # pragma: no cover
        """
        Reset agent to its initialization.
        :return:
        """
    