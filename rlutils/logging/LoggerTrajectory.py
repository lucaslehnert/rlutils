#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from rlutils.data import TransitionBuffer, TransitionListener


class LoggerTrajectory(TransitionListener):
    def __init__(self):
        self._curr_trajectory = TransitionBuffer()
        self._trajectory_list = []

    def update_transition(self, s, a, r, s_next, t, info):
        self._curr_trajectory.update_transition(s, a, r, s_next, t, info)
        if t:
            self.finish_trajectory()

    def get_trajectory_list(self):
        return self._trajectory_list

    def finish_trajectory(self):
        self._trajectory_list.append(self._curr_trajectory)
        self._curr_trajectory = TransitionBuffer()