#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestVIAgent(TestCase):
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

    def test_reset(self):
        import rlutils as rl
        import numpy as np

        t_mat, r_vec, q_corr, _, gamma = self._get_test_mdp()
        agent = rl.agent.VIAgent(t_mat, r_vec, gamma, "oh", eps=1e-5)
        states = [{"oh": rl.one_hot(i, 3)} for i in range(3)]
        q_vec = np.stack([agent.q_values(s) for s in states]).transpose()
        agent.update_transition(None, None, None, None, None, None)
        self.assertLessEqual(np.max(np.abs(q_vec - q_corr)), 1e-4)
        agent.reset()
        q_vec = np.stack([agent.q_values(s) for s in states]).transpose()
        self.assertLessEqual(np.max(np.abs(q_vec - q_corr)), 1e-4)

    def test_q_values(self):
        import rlutils as rl
        import numpy as np
        t_mat, r_vec, q_corr, v_corr, gamma = self._get_test_mdp()
        agent = rl.agent.VIAgent(t_mat, r_vec, gamma, "oh", eps=1e-5)
        states = [{"oh": rl.one_hot(i, 3)} for i in range(3)]
        q_vec = np.stack([agent.q_values(s) for s in states]).transpose()
        self.assertLessEqual(np.max(np.abs(q_vec - q_corr)), 1e-4)

    def test_update_transition(self):
        import rlutils as rl
        import numpy as np
        t_mat, r_vec, q_corr, v_corr, gamma = self._get_test_mdp()
        agent = rl.agent.VIAgent(t_mat, r_vec, gamma, "oh", eps=1e-5)
        agent.update_transition(None, None, None, None, None, None)
        states = [{"oh": rl.one_hot(i, 3)} for i in range(3)]
        q_vec = np.stack([agent.q_values(s) for s in states]).transpose()
        self.assertLessEqual(np.max(np.abs(q_vec - q_corr)), 1e-4)
