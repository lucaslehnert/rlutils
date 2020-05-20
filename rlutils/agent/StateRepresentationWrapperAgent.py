#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from .Agent import Agent


class StateRepresentationWrapperAgent(Agent):
    """
    Agent wrapper class that maps all state inputs through a specified state representation function.
    """

    def __init__(self, agent, phi=lambda s: s):
        """
        Wraps an agent using a state representation. If this wrapper is updated with a state s, then the wrapped agent
        will be updated with a state phi(s).

        :param agent: The agent that is wrapped.
        :param phi: A lambda mapping states to a different representation. The default is the identity function.
        """
        self._agent = agent
        self._phi = phi

    def get_abstract_agent(self):
        """

        :return: Wrapped agent.
        """
        return self._agent

    def reset(self, *params, **kwargs):
        self._agent.reset(*params, **kwargs)

    def q_values(self, state):
        return self._agent.q_values(self._phi(state))

    def update_transition(self, state, action, reward, next_state, term, info):
        return self._agent.update_transition(self._phi(state), action, reward, self._phi(next_state), term, info)
