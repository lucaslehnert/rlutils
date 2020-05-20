#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from .Agent import Agent


class ValueFunctionAgent(Agent):
    """
    ValueFunctionAgent can be used to construct an agent the follows a specified value function.

    The reset and update_transition methods have no effect. The q_values method returns Q-values using the value
    function that is passed as a constructor.
    """

    def __init__(self, q_fun):
        """
        An agent that wraps a value function. This agent will provide Q-values according to the provided value function.

        :param q_fun: A function mapping a state s to a vector of Q-values, where every i corresponds to the Q-value
            Q(s, i).
        """
        self._q_fun = q_fun

    def reset(self, *params, **kwargs):
        pass

    def q_values(self, state):
        return self._q_fun(state)

    def update_transition(self, state, action, reward, next_state, term, info):
        pass
