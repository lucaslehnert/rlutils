#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from .Policy import Policy


class ActionSequencePolicy(Policy):
    """
    A policy which will follow the provided action sequence. ActionSequenceTimeoutException once all actions are
    executed.
    """
    def __init__(self, action_sequence):
        self._action_sequence = action_sequence
        self._i = 0

    def __call__(self, state):
        if self._i >= len(self._action_sequence):
            raise ActionSequenceTimeoutException('Ran out of actions to select from action sequence.')
        act = self._action_sequence[self._i]
        self._i += 1
        return act

    def reset(self):
        """
        Reset execution of policy to the beginning of the action sequence.
        :return:
        """
        self._i = 0


class ActionSequenceTimeoutException(Exception):
    pass
