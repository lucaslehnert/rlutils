#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestUniformRandomPolicy(TestCase):

    def test(self):
        import rlutils as rl
        import numpy as np
        from scipy.stats import chisquare

        for seed in [1, 12, 123, 1234, 12345]:
            rl.set_seeds(seed)
            num_samples = 500
            num_actions = 4
            policy = rl.policy.uniform_random(num_actions)
            act_sample = np.array([policy(None) for _ in range(num_samples)])
            act_sample_cnts = np.array([np.sum(act_sample == i) for i in range(num_actions)])
            act_comp_cnts = (np.ones(num_actions) / num_actions) * num_samples
            p_val = chisquare(act_sample_cnts, act_comp_cnts).pvalue
            self.assertLessEqual(.1, p_val)
