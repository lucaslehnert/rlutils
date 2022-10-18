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

        mdp_recon = rl.environment.TabularMDP.load_from_file(
            'test/mdp_dir/mdp_meta.yaml'
        )
        t_mat, r_mat = mdp.get_t_mat_r_mat()
        t_mat_recon, r_mat_recon = mdp_recon.get_t_mat_r_mat()

        self.assertTrue(np.allclose(t_mat, t_mat_recon))
        self.assertTrue(np.allclose(r_mat, r_mat_recon))
        self.assertTrue(np.allclose(
            mdp.goal_state_list(), mdp_recon.goal_state_list()
        ))
        self.assertTrue(np.allclose(
            mdp.start_state_list(), mdp_recon.start_state_list()
        ))

        shutil.rmtree('test/mdp_dir')

    def test_state_defaults(self):
        import rlutils as rl
        import numpy as np

        mdp = rl.environment.PuddleWorld()
        state_defaults = mdp.state_defaults()
        self.assertTrue(np.allclose(
            state_defaults[rl.environment.PuddleWorld.ONE_HOT],
            np.zeros(100, dtype=np.float32)
        ))
        self.assertTrue(np.allclose(
            state_defaults[rl.environment.PuddleWorld.IDX],
            np.int32(-1)
        ))
        self.assertTrue(np.allclose(
            state_defaults[rl.environment.PuddleWorld.X],
            np.float32(-1)
        ))
        self.assertTrue(np.allclose(
            state_defaults[rl.environment.PuddleWorld.Y],
            np.float32(-1)
        ))

    def test_transition_defaults(self):
        import rlutils as rl
        import numpy as np

        mdp = rl.environment.PuddleWorld()
        transition_defaults = mdp.transition_defaults()
        self.assertTrue(np.allclose(
            transition_defaults[rl.data.Action],
            np.int32(-1)
        ))
        self.assertTrue(np.allclose(
            transition_defaults[rl.data.Reward],
            np.float32(0.)
        ))
        self.assertTrue(np.allclose(
            transition_defaults[rl.data.Term],
            False
        ))

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
        mdp = rl.environment.TabularMDP(
            t_mat, r_mat, idx_start_list=[0], idx_goal_list=[9]
        )

        t_mat_0, r_mat_0 = mdp.get_t_mat_r_mat()
        self.assertTrue(np.allclose(t_mat, t_mat_0))
        self.assertTrue(np.allclose(r_mat, r_mat_0))
        self.assertLessEqual(
            np.max(np.abs(np.sum(t_mat_0, axis=-1) - 1.)), 1e-5
        )

        t_mat_0, r_vec_0 = mdp.get_t_mat_r_vec()
        self.assertTrue(np.allclose(t_mat, t_mat_0))
        self.assertTrue(np.allclose(r_vec, r_vec_0))
        self.assertLessEqual(
            np.max(np.abs(np.sum(t_mat_0, axis=-1) - 1.)), 1e-5)

        self.assertEqual(np.shape(t_mat)[0], mdp.num_actions())
        self.assertEqual(np.shape(t_mat)[1], mdp.num_states())

    def test_start_and_goal_states(self):
        import rlutils as rl
        import numpy as np
        from itertools import product

        t_mat, r_mat, r_vec = self._get_rand_mdp_mat()
        start_list = [0, 1]
        goal_list = [4, 5, 6]
        mdp = rl.environment.TabularMDP(
            t_mat, r_mat, idx_start_list=start_list, idx_goal_list=goal_list
        )

        self.assertEqual(set(start_list), set(mdp.start_state_list()))
        self.assertEqual(set(goal_list), set(mdp.goal_state_list()))

        t_mat_mod, _ = mdp.get_t_mat_r_mat()
        for a, s in product(range(np.shape(t_mat)[0]), goal_list):
            t_mat_mod[a, s, s] = 1.
        self.assertLessEqual(
            np.max(np.abs(np.sum(t_mat_mod, axis=-1) - 1.)), 1e-5)

    def test_puddle_world_simulation(self):
        """
        This test is implemented in test/data/test_simulate.py in method 
        TestSimulate.test_simulate().
        """
        pass
