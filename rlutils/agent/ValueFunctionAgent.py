#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
from typing import Callable, Dict, Any
from .Agent import Agent


class ValueFunctionAgent(Agent):
    """
    ValueFunctionAgent can be used to construct an agent the follows a specified 
    value function.

    The reset and update_transition methods have no effect. The q_values method 
    returns Q-values using the value function that is passed as a constructor.
    """

    def __init__(self, q_fun: Callable[[Dict[str, Any]], np.ndarray]):
        """An agent that wraps a value function. This agent will provide 
        Q-values according to the provided value function.

        Args:
            q_fun (Callable[[Dict[str, Any]], np.ndarray]): Q-function
        """
        self._q_fun = q_fun

    def reset(self, *params, **kwargs):
        pass

    def q_values(self, state: Dict[str, Any]):
        return self._q_fun(state)

    def update_transition(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        reward: Any, 
        next_state: Dict[str, Any], 
        term: bool, 
        info: Dict[Any, Any]
    ):
        pass

    def on_simulation_timeout(self):
        pass

