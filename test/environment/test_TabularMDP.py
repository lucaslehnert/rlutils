#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import unittest


class TestPersistenceTabularMDP(unittest.TestCase):
    def test_persistance(self):
        import rlutils as rl
        import numpy as np
        import shutil

        mdp = rl.environment.PuddleWorld()
        mdp.save_to_file('test/mdp_dir/mdp_meta.yaml')

        mdp_recon = rl.environment.TabularMDP.load_from_file('test/mdp_dir/mdp_meta.yaml')
        t_mat, r_mat = mdp.get_t_mat_r_mat()
        t_mat_recon, r_mat_recon = mdp_recon.get_t_mat_r_mat()

        self.assertTrue(np.allclose(t_mat, t_mat_recon))
        self.assertTrue(np.allclose(r_mat, r_mat_recon))
        self.assertTrue(np.allclose(mdp.goal_state_list(), mdp_recon.goal_state_list()))
        self.assertTrue(np.allclose(mdp.start_state_list(), mdp_recon.start_state_list()))

        shutil.rmtree('test/mdp_dir')

    def test_name(self):
        import rlutils as rl
        self.assertEqual(str(rl.environment.PuddleWorld()), 'PuddleWorld')

    def _get_rand_mdp_mat(self):
        import numpy as np

        t_mat = np.random.uniform(size=[3, 10, 10])
        t_mat = t_mat / np.sum(t_mat, axis=-1, keepdims=True)
        r_mat = np.random.randint(0, 1, size=[3, 10, 10]).astype(np.float32)
        r_vec = np.sum(t_mat * r_mat, axis=-1)

        return t_mat, r_mat, r_vec

    def test_matrix_extraction(self):
        import rlutils as rl
        import numpy as np

        t_mat, r_mat, r_vec = self._get_rand_mdp_mat()
        mdp = rl.environment.TabularMDP(t_mat, r_mat, idx_start_list=[0], idx_goal_list=[9])

        t_mat_0, r_mat_0 = mdp.get_t_mat_r_mat()
        self.assertTrue(np.allclose(t_mat, t_mat_0))
        self.assertTrue(np.allclose(r_mat, r_mat_0))
        self.assertLessEqual(np.max(np.abs(np.sum(t_mat_0, axis=-1) - 1.)), 1e-5)

        t_mat_0, r_vec_0 = mdp.get_t_mat_r_vec()
        self.assertTrue(np.allclose(t_mat, t_mat_0))
        self.assertTrue(np.allclose(r_vec, r_vec_0))
        self.assertLessEqual(np.max(np.abs(np.sum(t_mat_0, axis=-1) - 1.)), 1e-5)

        self.assertEqual(np.shape(t_mat)[0], mdp.num_actions())
        self.assertEqual(np.shape(t_mat)[1], mdp.num_states())

    def test_start_and_goal_states(self):
        import rlutils as rl
        import numpy as np
        from itertools import product

        t_mat, r_mat, r_vec = self._get_rand_mdp_mat()
        start_list = [0, 1]
        goal_list = [4, 5, 6]
        mdp = rl.environment.TabularMDP(t_mat, r_mat, idx_start_list=start_list, idx_goal_list=goal_list)

        self.assertEqual(set(start_list), set(mdp.start_state_list()))
        self.assertEqual(set(goal_list), set(mdp.goal_state_list()))

        t_mat_mod, _ = mdp.get_t_mat_r_mat()
        for a, s in product(range(np.shape(t_mat)[0]), goal_list):
            t_mat_mod[a, s, s] = 1.
        self.assertLessEqual(np.max(np.abs(np.sum(t_mat_mod, axis=-1) - 1.)), 1e-5)

    def test_puddle_world_simulation(self):
        import rlutils as rl
        import numpy as np

        mdp = rl.environment.PuddleWorld(slip_prob=0.)
        traj = rl.data.TransitionBuffer()
        s = mdp.reset()
        action_seq = [
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.RIGHT,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.LEFT,
            rl.environment.gridworld.GridWorldAction.LEFT
        ]
        xy_seq = [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 9),
            (0, 9),
            (0, 9)
        ]
        s_seq_corr = [rl.environment.gridworld.pt_to_idx(xy, (10, 10)) for xy in xy_seq]
        for a in action_seq:
            sn, r, t, i = mdp.step(a)
            traj.update_transition(s, a, r, sn, t, i)
            s = sn
        self.assertTrue(np.all(np.array([0., 0., -1., -1., -1., -1., -1., -1., -1., 0., 0., 1., 0.]) == traj.all()[2]))
        self.assertTrue(np.all(np.where(traj.all()[0] == 1.)[1] == s_seq_corr[:-1]))
        self.assertTrue(np.all(np.where(traj.all()[3] == 1.)[1] == s_seq_corr[1:]))
        self.assertTrue(np.all(traj.all()[1] == action_seq))
        self.assertTrue(np.all(traj.all()[4] == [False] * 11 + [True] * 2))


