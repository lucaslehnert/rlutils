#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
from typing import Dict, Any
from .ValuePolicy import ValuePolicy


class GreedyPolicy(ValuePolicy):
    """
    Implementation of a greedy policy.

    """
    def _select_action(self, _: Dict[str, Any], q_values: np.ndarray) -> int:
        opt_act = np.where(np.max(q_values) == q_values)[0]
        return np.random.choice(opt_act)
