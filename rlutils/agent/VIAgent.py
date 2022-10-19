#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np

from .. import algorithm
from .ValueFunctionAgent import ValueFunctionAgent


class VIAgent(ValueFunctionAgent):
    """
    VIAgent performs value iteration on a tabular MDP when the object is 
    constructed. These Q-values are then subsequently used by the agent 
    subsequently. The reset and update_transition methods have no effect. The 
    q_values method returns Q-values that were computed using value iteration.

    States have to be represented as one-hot bit vectors.
    """

    def __init__(
        self, 
        t_mat: np.ndarray, 
        r_vec: np.ndarray, 
        gamma: float, 
        state_column_name: str, 
        eps: float=1e-5, 
        max_it: int=5000
    ):
        """An agent that uses a full transition matrix and reward vector to run 
        value iteration and then use the computed value function for decision 
        making.

        Args:
            t_mat (np.ndarray): _description_
            r_vec (np.ndarray): _description_
            gamma (float): _description_
            state_column_name (str): _description_
            eps (_type_, optional): _description_. Defaults to 1e-5.
            max_it (int, optional): _description_. Defaults to 5000.
        """
        self._q, _ = algorithm.vi(t_mat, r_vec, gamma, eps=eps, max_it=max_it)
        super().__init__(
            q_fun=lambda s: np.matmul(self._q, s[state_column_name]))

