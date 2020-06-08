#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestValueFunctionAgent(TestCase):
    def test_reset(self):
        import rlutils as rl
        import numpy as np
        agent = rl.agent.ValueFunctionAgent(q_fun=lambda s: np.ones(3) * 0.4)
        agent.reset()
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 2)) == 0.4))

    def test_q_values(self):
        import rlutils as rl
        import numpy as np
        agent = rl.agent.ValueFunctionAgent(q_fun=lambda s: np.ones(3) * 0.4)
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 2)) == 0.4))
        agent.on_simulation_timeout()
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 2)) == 0.4))

    def test_update_transition(self):
        import rlutils as rl
        import numpy as np
        agent = rl.agent.ValueFunctionAgent(q_fun=lambda s: np.ones(3) * 0.4)
        agent.update_transition(None, None, None, None, None, None)
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 2)) == 0.4))
