#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestLinearInterpolatedVariableSchedule(TestCase):
    def test(self):
        import rlutils as rl
        schedule = rl.schedule.LinearInterpolatedVariableSchedule([0, 1], [0, 1])
        self.assertEqual(schedule(0), 0.)
        self.assertEqual(schedule(1), 1.)
        self.assertEqual(schedule(.5), .5)
        self.assertEqual(schedule(.75), .75)
        self.assertEqual(schedule(.25), .25)
        self.assertEqual(schedule(2), 1.)
        self.assertEqual(schedule.get_t_list()[0], 0)
        self.assertEqual(schedule.get_t_list()[1], 1)
        self.assertEqual(schedule.get_v_list()[0], 0)
        self.assertEqual(schedule.get_v_list()[1], 1)
        