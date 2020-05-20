#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestGreedyPolicy(TestCase):
    def test(self):
        import rlutils as rl
        import numpy as np
        from scipy.stats import chisquare

        for seed in [1, 12, 123, 1234, 12345]:
            rl.set_seeds(seed)
            q_vals = np.array([
                [0, 1],
                [0, 1],
                [1, 0]
            ], dtype=np.float32)
            agent = rl.agent.ValueFunctionAgent(q_fun=lambda s: np.matmul(q_vals, s))
            policy = rl.policy.GreedyPolicy(agent)
            act_0 = np.array([policy(rl.one_hot(0, 2)) for _ in range(100)])
            self.assertTrue(np.all(act_0 == 2))
            act_1 = np.array([policy(rl.one_hot(1, 2)) for _ in range(100)])
            self.assertTrue(np.all([a in [0, 1] for a in act_1]))
            act_1_cnts = np.array([np.sum(act_1 == i) for i in [0, 1]])
            p_val = chisquare(act_1_cnts, [50, 50]).pvalue
            self.assertLessEqual(.1, p_val)
