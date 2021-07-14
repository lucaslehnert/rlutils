#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestEGreedyPolicy(TestCase):
    def test(self):
        import rlutils as rl
        import numpy as np
        from scipy.stats import chisquare
        from itertools import product

        it = product(
            [1, 12, 123, 1234, 12345], 
            [0., .25, .5, .75, 1.], 
            [0, 1]
        )
        for (seed, eps, state) in it:
            rl.set_seeds(seed)
            q_vals = np.array([
                [0, 1],
                [0, 1],
                [1, 0]
            ], dtype=np.float32)
            agent = rl.agent.ValueFunctionAgent(
                q_fun=lambda s: np.matmul(q_vals, s)
            )

            prob_greedy = np.array(q_vals, copy=True) / np.sum(q_vals, axis=0)
            prob_uniform = np.ones(np.shape(q_vals)) / 3.
            prob = (1. - eps) * prob_greedy + eps * prob_uniform
            policy = rl.policy.EGreedyPolicy(agent, eps)

            act = np.array([policy(rl.one_hot(state, 2)) for _ in range(100)])
            if eps == 0. and state == 0:
                self.assertTrue(np.all([a == 2 for a in act]))
            elif eps == 0. and state == 1:
                self.assertTrue(np.all([a in [0, 1] for a in act]))
            else:
                self.assertTrue(np.all([a in [0, 1, 2] for a in act]))

            act_cnts = np.array([np.sum(act == i) for i in range(3)])
            p_val = chisquare(act_cnts, prob[:, state] * 100. + 1e-10).pvalue
            print(p_val)
            self.assertLessEqual(.01, p_val)
