#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import random
from typing import Dict, Any
from ..types import Policy


class UniformRandomPolicy(Policy):
    """
    A policy which selects actions uniformly at random.
    """
    def __init__(self, num_actions: int):
        self._num_actions = num_actions

    def __call__(self, _: Dict[str, Any]) -> int:
        return random.randint(0, self._num_actions - 1)

def uniform_random(num_actions: int) -> UniformRandomPolicy:
    """Returns a policy selecting actions uniformly at random.

    Args:
        num_actions (int): Number of actions.

    Returns:
        UniformRandomPolicy: An object that implements Policy interface.
    """
    return UniformRandomPolicy(num_actions)
