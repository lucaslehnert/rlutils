#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import unittest


class TestVi(unittest.TestCase):
    '''
    Test rlutils.algorithm.vi.
    '''

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

        return t_mat, r_vec, q_corr, v_corr

    def test_vi(self):
        import rlutils as rl
        import numpy as np

        t_mat, r_vec, q_corr, v_corr = self._get_test_mdp()

        q_hat, v_hat = rl.algorithm.vi(t_mat, r_vec, gamma=0.9, eps=1e-5)
        self.assertLess(np.max(np.abs(v_hat - v_corr)), 1e-4)
        self.assertLess(np.max(np.abs(q_hat - q_corr)), 1e-4)

    def test_vi_q_init(self):
        import rlutils as rl
        import numpy as np

        t_mat, r_vec, q_corr, v_corr = self._get_test_mdp()

        q_hat, v_hat = rl.algorithm.vi(t_mat, r_vec, gamma=0.9, eps=1e-5, q_init=q_corr, max_it=1)
        self.assertLess(np.max(np.abs(v_hat - v_corr)), 1e-4)
        self.assertLess(np.max(np.abs(q_hat - q_corr)), 1e-4)

    def test_vi_v_init(self):
        import rlutils as rl
        import numpy as np

        t_mat, r_vec, q_corr, v_corr = self._get_test_mdp()

        q_hat, v_hat = rl.algorithm.vi(t_mat, r_vec, gamma=0.9, eps=1e-5, v_init=v_corr, max_it=1)
        self.assertLess(np.max(np.abs(v_hat - v_corr)), 1e-4)
        self.assertLess(np.max(np.abs(q_hat - q_corr)), 1e-4)


    def test_vi_timeout(self):
        import rlutils as rl

        t_mat, r_vec, q_corr, v_corr = self._get_test_mdp()

        try:
            rl.algorithm.vi(t_mat, r_vec, gamma=0.9, eps=1e-5, max_it=2)
            self.fail()
        except rl.algorithm.VITimeoutException:
            pass


if __name__ == '__main__':
    unittest.main()
