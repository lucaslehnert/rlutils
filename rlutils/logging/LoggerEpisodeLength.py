#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np

from rlutils.data import TransitionListener


class LoggerEpisodeLength(TransitionListener):
    """
    Logger class that logs the episode length.
    """

    def __init__(self):
        self._curr_episode_steps = 0
        self._episode_length = []

    def update_transition(self, s, a, r, s_next, t, info):
        self._curr_episode_steps += 1
        if t:
            self.finish_trajectory()

    def get_episode_length(self):
        return np.array(self._episode_length)

    def update_episode_length(self, episode_length):
        self._episode_length.append(episode_length)
        self._curr_episode_steps = 0

    def finish_trajectory(self):
        self._episode_length.append(self._curr_episode_steps)
        self._curr_episode_steps = 0
