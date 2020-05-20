#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np


def vi(t_mat, r_vec,
       gamma=.99,
       eps=1e-5,
       max_it=5000,
       target_op=lambda q: np.max(q, axis=0),
       q_init=None,
       v_init=None):
    """
    Implementation of value iteration.

    :param t_mat: A transition table of the shape [num_actions, num_states, num_states]
    :param r_vec: A reward table of the shape [num_actions, num_states, num_states]
    :param gamma: Discount factor.
    :param eps: A small epsilon. If an iteration yields changes that fall below eps, the algorithm terminates.
    :param max_it: Maximum iteration threshold. If this threshold is reached, the algorithm raises a VITimeoutException.
    :param target_op: Target operation, by default the maximum Q-value is used. This operation could be changed to
        evaluate a particular policy.
    :param q_init: Initial Q-values. If None, then zero Q-values will be used.
    :param v_init: If None, initialization is computed from Q-values.
    :return: q, v
        q: A Q-value table of size [num_actions, num_states].
        v: A state value table of size [num_states].
    """
    num_actions, num_states = np.shape(r_vec)
    if q_init is None:
        q = np.zeros([num_actions, num_states])
    else:
        q = q_init
    if v_init is None:
        v = target_op(q)
    else:
        v = v_init

    converged = False
    it = 0
    while not converged:
        for a in range(num_actions):
            q[a] = r_vec[a] + gamma * np.matmul(t_mat[a], v)

        v_next = target_op(q)
        td_error = np.linalg.norm(v_next - v, ord=np.inf)
        converged = td_error < eps
        v = v_next
        it += 1
        if max_it is not None and it > max_it:
            raise VITimeoutException('VI time out, TD error: %1.5e' % td_error)
    return q, v


class VITimeoutException(Exception):
    pass
