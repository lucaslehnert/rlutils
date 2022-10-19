#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestGridworld(TestCase):
    def test_generate_mdp_from_transition_and_reward_function(self):
        import rlutils as rl
        import numpy as np
        from rlutils.environment.gridworld import generate_mdp_from_transition_and_reward_function

        def t_fn(s, a, sn):
            if a == 0 and s == 0 and sn == 1:
                return 1.
            elif a == 1 and s == 0 and sn == 0:
                return 1.
            elif a == 0 and s == 1 and sn == 1:
                return 1.
            elif a == 1 and s == 1 and sn == 0:
                return 1.
            else:
                return 0.

        def r_fn(s, a, sn):
            if a == 0 and s == 1 and sn == 1:
                return 1.
            else:
                return 0.

        t_mat_corr = np.array([
            [
                [0, 1],
                [0, 1]
            ],
            [
                [1, 0],
                [1, 0]
            ]
        ], dtype=np.float32)
        r_mat_corr = np.array([
            [
                [0, 0],
                [0, 1]
            ],
            [
                [0, 0],
                [0, 0]
            ]
        ], dtype=np.float32)
        r_vec_corr = np.sum(t_mat_corr * r_mat_corr, axis=-1)

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(
            num_states=2,
            num_actions=2,
            transition_fn=t_fn,
            reward_fn=r_fn,
            reward_matrix=True
        )
        self.assertTrue(np.allclose(t_mat, t_mat_corr))
        self.assertTrue(np.allclose(r_mat, r_mat_corr))

        t_mat, r_vec = generate_mdp_from_transition_and_reward_function(
            num_states=2,
            num_actions=2,
            transition_fn=t_fn,
            reward_fn=r_fn,
            reward_matrix=False
        )
        self.assertTrue(np.allclose(t_mat, t_mat_corr))
        self.assertTrue(np.allclose(r_vec, r_vec_corr))

    def test_generate_gridworld_transition_function_with_barrier(self):
        from rlutils.environment.gridworld import generate_gridworld_transition_function_with_barrier
        from rlutils.environment.gridworld import generate_transition_matrix_from_function
        import numpy as np

        t_mat_corr = np.stack([np.eye(2)] * 4)
        t_fn = generate_gridworld_transition_function_with_barrier(
            2, 1, slip_prob=0., barrier_idx_list=[(0, 1)])
        t_mat = generate_transition_matrix_from_function(2, 4, t_fn)
        self.assertTrue(np.allclose(t_mat, t_mat_corr))

        t_fn = generate_gridworld_transition_function_with_barrier(
            1, 2, slip_prob=0., barrier_idx_list=[(0, 1)])
        t_mat = generate_transition_matrix_from_function(2, 4, t_fn)
        self.assertTrue(np.allclose(t_mat, t_mat_corr))

    def test_idx_to_pt(self):
        from rlutils.environment.gridworld import idx_to_pt
        self.assertEqual(idx_to_pt(0, (2, 3)), (0, 0))
        self.assertEqual(idx_to_pt(1, (2, 3)), (1, 0))
        self.assertEqual(idx_to_pt(2, (2, 3)), (0, 1))
        self.assertEqual(idx_to_pt(3, (2, 3)), (1, 1))
        self.assertEqual(idx_to_pt(4, (2, 3)), (0, 2))
        self.assertEqual(idx_to_pt(5, (2, 3)), (1, 2))

    def test_pt_to_idx(self):
        from rlutils.environment.gridworld import pt_to_idx
        self.assertEqual(pt_to_idx((0, 0), (2, 3)), 0)
        self.assertEqual(pt_to_idx((1, 0), (2, 3)), 1)
        self.assertEqual(pt_to_idx((0, 1), (2, 3)), 2)
        self.assertEqual(pt_to_idx((1, 1), (2, 3)), 3)
        self.assertEqual(pt_to_idx((0, 2), (2, 3)), 4)
        self.assertEqual(pt_to_idx((1, 2), (2, 3)), 5)
