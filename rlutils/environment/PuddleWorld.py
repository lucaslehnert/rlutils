#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np

from .TabularMDP import TabularMDP
from .gridworld import generate_gridworld_transition_function
from .gridworld import generate_mdp_from_transition_and_reward_function
from .gridworld import pt_to_idx, idx_to_pt


class PuddleWorld(TabularMDP):
    X = 'x'
    Y = 'y'

    def __init__(self, slip_prob=0.05, dtype=np.float32):
        start = (0, 0)
        goal = (0, 9)

        t_fn = generate_gridworld_transition_function(10, 10, slip_prob=slip_prob)

        def r_fn(s_1, a, s_2):
            x, y = idx_to_pt(s_2, (10, 10))
            goal_rew = 0.
            if goal[0] == x and goal[1] == y:
                goal_rew = 1.0
            puddle_penalty = 0.
            if 0 <= x <= 5 and 2 <= y <= 8:
                puddle_penalty = -1.0
            return goal_rew + puddle_penalty

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=dtype)
        super().__init__(t_mat, r_mat, [pt_to_idx(start, (10, 10))], [pt_to_idx(goal, (10, 10))], name='PuddleWorld')

    def _augment_state_dict(self, s: dict) -> dict:
        i = np.where(s[TabularMDP.ONE_HOT] == 1)[0][0]
        x, y = idx_to_pt(i, (10, 10))
        s[PuddleWorld.X] = x
        s[PuddleWorld.Y] = y
        return s

    def reset(self, *params, **kwargs):
        return self._augment_state_dict(super().reset(*params, **kwargs))
    
    def step(self, action):
        s_n, r, t, _ = super().step(action)
        if r == 1.:
            info = {'puddle': False, 'goal': True}
        elif r == -1:
            info = {'puddle': True, 'goal': False}
        else:
            info = {'puddle': False, 'goal': False}
        return self._augment_state_dict(s_n), r, t, info
