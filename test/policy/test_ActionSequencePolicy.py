#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestActionSequencePolicy(TestCase):
    def test(self):
        import rlutils as rl
        act_seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        policy = rl.policy.ActionSequencePolicy(act_seq)
        for a in act_seq:
            self.assertEqual(policy(None), a)

        try:
            policy(None)
            self.fail()
        except rl.policy.ActionSequenceTimeoutException:
            pass

        policy.reset()
        for a in act_seq:
            self.assertEqual(policy(None), a)
