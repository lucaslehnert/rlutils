#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
from rlutils.data import TransitionListener


class LoggerTotalReward(TransitionListener):
    """
    Logger class that logs the total reward incurred per episode.
    """

    def __init__(self):
        self._total_reward = 0
        self._total_reward_episodic = []

    def update_transition(self, s, a, r, s_next, t, info):
        self._total_reward += r
        if t:
            self._total_reward_episodic.append(self._total_reward)
            self._total_reward = 0

    def get_total_reward_episodic(self):
        return np.array(self._total_reward_episodic)

    def finish_episode(self):
        '''
        Call this if an episode times out or does not finish with a terminal flag.

        :return:
        '''
        self._total_reward_episodic.append(self._total_reward)
        self._total_reward = 0
