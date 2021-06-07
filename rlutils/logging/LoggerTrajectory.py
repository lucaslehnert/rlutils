#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from rlutils.data import TransitionListener, ReplayBuffer, ACTION, REWARD, TERM


class LoggerTrajectory(TransitionListener):
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer

    def update_transition(self, s, a, r, s_next, t, info):
        self.replay_buffer.add_transition(
            state=s,
            transition={
                ACTION: a,
                REWARD: r,
                TERM: t
            },
            next_state=s_next
        )
        if t:
            self.replay_buffer.finish_current_sequence()

    def on_simulation_timeout(self):
        self.replay_buffer.finish_current_sequence()