#
# Copyright (c) 2021 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in 
# the root directory of this project.
#

from unittest import TestCase
import rlutils as rl

class StateStoringAgent(rl.agent.Agent):
    def __init__(self):
        super(StateStoringAgent, self).__init__()
        self.state = None
        self.next_state = None

    def on_simulation_timeout(self):
        pass

    def reset(self, *params, **kwargs):
        self.state = None
        self.next_state = None

    def q_values(self, state):
        self.state = state
        return None
        
    def update_transition(self, state, action, reward, next_state, term, info):
        self.state = state
        self.next_state = next_state


class TestStateBatchWrapperAgent(TestCase):
    def test_q_values(self):
        import rlutils as rl
        import numpy as np

        agent = StateStoringAgent()
        agent_wrapped = rl.agent.StateBatchWrapperAgent(agent)

        s = np.zeros([5, 5, 1], dtype=np.int32)
        agent_wrapped.q_values(s)
        self.assertTrue(np.all(agent.state == np.array([s])))

    def test_transition(self):
        import rlutils as rl
        import numpy as np

        agent = StateStoringAgent()
        agent_wrapped = rl.agent.StateBatchWrapperAgent(agent)

        s = np.zeros([5, 5, 1], dtype=np.float32)
        next_s = np.ones([5, 5, 1], dtype=np.float32)
        agent_wrapped.update_transition(
            s, 0, 1., next_s, False, None
        )

        self.assertTrue(np.all(agent.state == np.array([s])))
        self.assertTrue(np.all(agent.next_state == np.array([next_s])))
