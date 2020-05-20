#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np


def generate_transition_matrix_from_function(num_states, num_actions, transition_fn, dtype=np.float64):
    t_mat = np.zeros([num_actions, num_states, num_states], dtype=dtype)

    for a in range(num_actions):
        for s_1 in range(num_states):
            for s_2 in range(num_states):
                t_mat[a, s_1, s_2] = transition_fn(s_1, a, s_2)
    return t_mat


generate_reward_matrix_from_function = generate_transition_matrix_from_function


def generate_mdp_from_transition_and_reward_function(num_states, num_actions, transition_fn, reward_fn,
                                                     reward_matrix=False,
                                                     dtype=np.float64):
    t_mat = generate_transition_matrix_from_function(num_states, num_actions, transition_fn, dtype=dtype)
    r_mat = generate_reward_matrix_from_function(num_states, num_actions, reward_fn, dtype=dtype)

    if reward_matrix:
        return t_mat, r_mat
    else:
        r_vec = np.sum(t_mat * r_mat, axis=-1)
        return t_mat, r_vec


def add_terminal_states(t_mat, r_mat, term_state_mask):
    num_actions, num_states, _ = np.shape(t_mat)
    for i, t in enumerate(term_state_mask):
        for a in range(num_actions):
            if t:
                t_mat[a, i, :] *= 0.
                t_mat[a, i, i] = 1.
                r_mat[a, i, :] *= 0.
    return t_mat, r_mat


def idx_to_pt(i, shape):
    w, h = shape
    y = np.floor(i / w).astype(np.int)
    x = i - y * w
    return x, y


def pt_to_idx(pt, shape):
    w, h = shape
    x, y = pt
    return y * w + x


class GridWorldAction:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def generate_gridworld_transition_function(size_x, size_y, slip_prob=0.):
    '''
    Returns a grid world transition function. The origin of the coordinate system is in the top-left corner.

    :param size_x:
    :param size_y:
    :param slip_prob:
    :return:
    '''

    def t_fn(s, a, s_next):
        nonlocal size_x, size_y, slip_prob
        x, y = idx_to_pt(s, (size_x, size_y))

        if a == GridWorldAction.UP:
            x_next = x
            y_next = max(y - 1, 0)
        elif a == GridWorldAction.RIGHT:
            x_next = min(x + 1, size_x - 1)
            y_next = y
        elif a == GridWorldAction.DOWN:
            x_next = x
            y_next = min(y + 1, size_y - 1)
        elif a == GridWorldAction.LEFT:
            x_next = max(x - 1, 0)
            y_next = y
        s_next_pred = pt_to_idx((x_next, y_next), (size_x, size_y))

        if s_next == s and s_next == s_next_pred:
            return 1.0
        if not s_next == s and s_next == s_next_pred:
            return 1.0 - slip_prob
        elif s_next == s:
            return slip_prob
        else:
            return 0.0

    return t_fn


def generate_gridworld_transition_function_with_barrier(size_x, size_y, slip_prob=0., barrier_idx_list=[]):
    t_fn = generate_gridworld_transition_function(size_x, size_y, slip_prob)

    across_barrier = lambda s1, s2: (s1, s2) in barrier_idx_list or (s2, s1) in barrier_idx_list

    def other_side_states(s):
        nonlocal barrier_idx_list

        s_list = []
        for (x, y) in barrier_idx_list:
            if x == s:
                s_list.append(y)
            if y == s:
                s_list.append(x)
        return s_list

    def t_fn_barrier(s, a, s_next):
        nonlocal t_fn, barrier_idx_list
        if across_barrier(s, s_next):
            return 0.
        elif s == s_next and np.any([t_fn(s, a, s_o) > 0. for s_o in other_side_states(s)]):
            return 1.
        else:
            return t_fn(s, a, s_next)

    return t_fn_barrier
