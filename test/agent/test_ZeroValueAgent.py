#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestZeroValueAgent(TestCase):
    def test_q_values(self):
        import rlutils as rl
        import numpy as np
        agent = rl.agent.ZeroValueAgent(3)
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 1)) == 0.))

    def test_reset(self):
        import rlutils as rl
        import numpy as np
        agent = rl.agent.ZeroValueAgent(3)
        agent.reset()
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 1)) == 0.))

    def test_update_transition(self):
        import rlutils as rl
        import numpy as np
        agent = rl.agent.ZeroValueAgent(3)
        agent.update_transition(None, None, None, None, None, None)
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 1)) == 0.))


class TestUniformActionSelectionAgent(TestCase):
    def test_q_values(self):
        import rlutils as rl
        import numpy as np
        agent = rl.agent.UniformActionSelectionAgent(3)
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 1)) == 0.))

    def test_reset(self):
        import rlutils as rl
        import numpy as np
        agent = rl.agent.UniformActionSelectionAgent(3)
        agent.update_transition(None, None, None, None, None, None)
        agent.reset()
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 1)) == 0.))

    def test_update_transition(self):
        import rlutils as rl
        import numpy as np
        agent = rl.agent.UniformActionSelectionAgent(3)
        agent.update_transition(None, None, None, None, None, None)
        self.assertTrue(np.all(agent.q_values(rl.one_hot(0, 1)) == 0.))
