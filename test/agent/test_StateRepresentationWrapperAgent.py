#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in 
# the root directory of this project.
#

from unittest import TestCase


class TestStateRepresentationWrapperAgent(TestCase):
    def _get_test_mdp(self):
        import numpy as np

        t_mat = np.array([
            [
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 0., 1.]
            ],
            [
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.]
            ]
        ], dtype=np.float32)
        r_vec = np.array([
            [0., 0., 1.],
            [0., 0., 0.]
        ], dtype=np.float32)

        v_corr = np.array([10. * .9 * .9, 10. * .9, 10.], dtype=np.float32)
        q_corr = np.array([
            [10. * .9 * .9, 10. * .9, 10.],
            [10. * .9 * .9 * .9, 10. * .9 * .9 * .9, 10. * .9 * .9]
        ], dtype=np.float32)

        return t_mat, r_vec, q_corr, v_corr, 0.9

    def _get_representation_mat(self):
        import numpy as np
        phi_mat = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.]
        ], dtype=np.float32)
        return phi_mat

    def test_get_abstract_agent(self):
        import rlutils as rl
        import numpy as np
        t_mat, r_vec, q_corr, v_corr, gamma = self._get_test_mdp()
        rep_mat = self._get_representation_mat()
        agent_latent = rl.agent.VIAgent(t_mat, r_vec, gamma, eps=1e-5)
        agent = rl.agent.StateRepresentationWrapperAgent(
            agent_latent, lambda s: np.matmul(s, rep_mat)
            )
        self.assertEqual(agent_latent, agent.get_abstract_agent())

    def test_reset(self):
        import rlutils as rl
        import numpy as np
        rep_mat = self._get_representation_mat()
        agent_latent = rl.agent.QLearning(
            num_states=3, 
            num_actions=2, 
            learning_rate=0.1, 
            gamma=0.9, 
            init_Q=0.1
        )
        agent = rl.agent.StateRepresentationWrapperAgent(
            agent_latent, lambda s: np.matmul(s, rep_mat)
        )
        res = agent.update_transition(
            rl.one_hot(0, 9), 0, 0., rl.one_hot(1, 9), term=False, info={}
        )
        self.assertEqual(res['td_error'], .0 + .9 * .1 - .1)
        q_corr = np.ones([2, 3], dtype=np.float32) * .1
        q_corr[0, 0] = .1 + .1 * (.0 + .9 * .1 - .1)
        q_corr = np.stack([np.matmul(rep_mat, q_corr[a]) for a in range(2)])
        q_val_learned = np.stack(
            [agent.q_values(rl.one_hot(i, 9)) for i in range(9)]).transpose()
        self.assertTrue(np.allclose(q_corr, q_val_learned))
        agent.on_simulation_timeout()
        self.assertTrue(np.allclose(q_corr, q_val_learned))
        agent.reset()
        q_val_learned = np.stack(
            [agent.q_values(rl.one_hot(i, 9)) for i in range(9)]).transpose()
        self.assertTrue(np.all(q_val_learned == 0.1))

    def test_q_values(self):
        import rlutils as rl
        import numpy as np
        t_mat, r_vec, q_corr, v_corr, gamma = self._get_test_mdp()
        rep_mat = self._get_representation_mat()
        agent_latent = rl.agent.VIAgent(t_mat, r_vec, gamma, eps=1e-5)
        agent = rl.agent.StateRepresentationWrapperAgent(
            agent_latent, lambda s: np.matmul(s, rep_mat)
        )
        q_vec = np.stack(
            [agent.q_values(rl.one_hot(i, 9)) for i in range(9)]).transpose()
        q_infl = np.stack([np.matmul(rep_mat, q_corr[a]) for a in range(2)])
        self.assertLessEqual(np.max(np.abs(q_vec - q_infl)), 1e-4)

    def test_update_transition(self):
        import rlutils as rl
        import numpy as np
        rep_mat = self._get_representation_mat()
        agent_latent = rl.agent.QLearning(
            num_states=3, 
            num_actions=2, 
            learning_rate=0.1, 
            gamma=0.9, 
            init_Q=0.1
        )
        agent = rl.agent.StateRepresentationWrapperAgent(
            agent_latent, lambda s: np.matmul(s, rep_mat)
        )
        res = agent.update_transition(
            rl.one_hot(0, 9), 0, 0., rl.one_hot(1, 9), term=False, info={})
        self.assertEqual(res['td_error'], .0 + .9 * .1 - .1)
        q_corr = np.ones([2, 3], dtype=np.float32) * .1
        q_corr[0, 0] = .1 + .1 * (.0 + .9 * .1 - .1)
        q_corr = np.stack([np.matmul(rep_mat, q_corr[a]) for a in range(2)])
        q_val_learned = np.stack(
            [agent.q_values(rl.one_hot(i, 9)) for i in range(9)]).transpose()
        self.assertTrue(np.allclose(q_corr, q_val_learned))

