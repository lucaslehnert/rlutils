#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestActionTiledBasisFunction(TestCase):
    def test(self):
        import rlutils as rl
        import numpy as np

        num_s = 5
        num_a = 3
        basis_fn = rl.basisfunction.ActionTiledBasisFunction(num_s, num_a)
        for s in range(num_s):
            for a in range(num_a):
                s_vec = rl.one_hot(s, num_s)
                sa_vec = basis_fn(s_vec, a)
                self.assertEqual(np.where(sa_vec == 1.)[0][0], a * num_s + s)

