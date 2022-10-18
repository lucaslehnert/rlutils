#
# Copyright (c) 2021 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in 
# the root directory of this project.
#

# from .Agent import Agent
# import numpy as np


# class StateRepresentationWrapperAgent(Agent):
#     """
#     Agent wrapper class that maps all state inputs through a specified state 
#     representation function.
#     """

#     def __init__(self, agent: Agent, phi):
#         """
#         Wraps an agent using a state representation. If this wrapper is updated 
#         with a state s, then the wrapped agent will be updated with a state 
#         phi(s).

#         :param agent: The agent that is wrapped.
#         :param phi: A lambda mapping states to a different representation. 
#             The default is the identity function.
#         """
#         self._agent = agent
#         self._phi = phi

#     def get_abstract_agent(self):
#         """

#         :return: Wrapped agent.
#         """
#         return self._agent

#     def on_simulation_timeout(self):
#         self._agent.on_simulation_timeout()

#     def reset(self, *params, **kwargs):
#         self._agent.reset(*params, **kwargs)

#     def q_values(self, state):
#         return self._agent.q_values(self._phi(state))

#     def update_transition(self, state, action, reward, next_state, term, info):
#         state = self._phi(state)
#         next_state = self._phi(next_state)
#         return self._agent.update_transition(
#             state, action, reward, next_state, term, info
#         )


# class StateBatchWrapperAgent(StateRepresentationWrapperAgent):
#     def __init__(self, agent: Agent):
#         super(StateBatchWrapperAgent, self).__init__(
#             agent,
#             lambda s: np.array([s])
#         )

        
# class StateDictToNumpyWrapperAgent(StateRepresentationWrapperAgent):
#     def __init__(self, agent: Agent, dict_key: str):
#         super(StateDictToNumpyWrapperAgent, self).__init__(
#             agent,
#             lambda s: s[dict_key]
#         )
