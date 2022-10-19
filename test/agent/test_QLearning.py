#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestQLearning(TestCase):
    def test_reset(self):
        import rlutils as rl
        import numpy as np

        agent = rl.agent.QLearning(
            state_column_name="s", 
            num_states=2, 
            num_actions=2, 
            learning_rate=0.1, 
            gamma=0.9, 
            init_Q=0.1
        )
        agent.update_transition({"s": 0}, 0, 0., {"s": 1}, term=False, info={})
        agent.reset()
        self.assertTrue(np.all(agent.q_vector == 0.1))
        agent.reset(q_vec=np.ones((2, 2)) * 0.2)
        self.assertTrue(np.all(agent.q_vector == 0.2))

    def test_q_values(self):
        import rlutils as rl
        import numpy as np
        
        agent = rl.agent.QLearning(
            state_column_name="s", 
            num_states=2, 
            num_actions=2, 
            learning_rate=0.1, 
            gamma=0.9, 
            init_Q=0.1
        )
        self.assertTrue(np.all(agent.q_values({"s": 0}) == 0.1))
        self.assertTrue(np.all(agent.q_values({"s": 1}) == 0.1))

    def test_update_transition(self):
        import rlutils as rl
        import numpy as np

        agent = rl.agent.QLearning(
            state_column_name="s", 
            num_states=2, 
            num_actions=2, 
            learning_rate=0.1, 
            gamma=0.9, 
            init_Q=0.1
        )
        agent.update_transition({"s": 0}, 0, 0., {"s": 1}, term=False, info={})
        q_0 = np.array([.1 + .1 * (.0 + .9 * .1 - .1), .1])
        self.assertTrue(np.allclose(q_0, agent.q_values({"s": 0})))
        agent.on_simulation_timeout()
        self.assertTrue(np.allclose(q_0, agent.q_values({"s": 0})))
        q_1 = np.array([.1, .1])
        self.assertTrue(np.allclose(q_1, agent.q_values({"s": 1})))

    def test_q_vector(self):
        import rlutils as rl
        import numpy as np

        agent = rl.agent.QLearning(
            state_column_name="s", 
            num_states=2, 
            num_actions=2, 
            learning_rate=0.1, 
            gamma=0.9, 
            init_Q=0.1
        )
        self.assertTrue(np.all(agent.q_vector == 0.1))
        agent = rl.agent.QLearning(
            state_column_name="s", 
            num_states=2, 
            num_actions=2, 
            learning_rate=0.1, 
            gamma=0.9, 
            init_Q=np.ones((2, 2)) * .3
        )
        self.assertTrue(np.all(agent.q_vector == 0.3))

    def test_get_gamma(self):
        import rlutils as rl

        agent = rl.agent.QLearning(
            state_column_name="s", 
            num_states=2, 
            num_actions=2, 
            learning_rate=0.1, 
            gamma=0.9, 
            init_Q=None
        )
        self.assertEqual(agent.gamma, 0.9)

    def test_get_learning_rate(self):
        import rlutils as rl
        agent = rl.agent.QLearning(
            state_column_name="s", 
            num_states=2, 
            num_actions=2, 
            learning_rate=0.1, 
            gamma=0.9, 
            init_Q=None
        )
        self.assertEqual(agent.learning_rate, 0.1)

