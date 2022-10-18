#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np


class ActionTiledBasisFunction:
    """
    ActionTiledBasisFunction implements a tiled basis function for finite action 
    spaces. Each state is represented as a vector v. All actions are indices 
    ranging from 0 to num_actions - 1. The resulting basis function vector has a
    length of len(v) * num_actions. For an a, the resulting tiled vector w will 
    contain v at w[a * num_actions:(a+1) * num_actions] = v and will be zero 
    otherwise.
    """
    def __init__(self, state_dim: int, num_actions: int):
        """

        :param state_dim:
        :param num_actions:
        """
        self._state_dim = state_dim
        self._num_actions = num_actions

    def __call__(self, s_phi, action):
        sa_phi = np.zeros(self._state_dim * self._num_actions, dtype=np.float32)
        sa_phi[action * self._state_dim:(action + 1) * self._state_dim] = s_phi
        return sa_phi
