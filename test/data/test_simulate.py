#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestSimulate(TestCase):
    def test_simulate(self):
        import rlutils as rl
        import numpy as np
        logger = rl.logging.LoggerEpisodeLength()
        for _ in range(5):
            mdp = rl.environment.PuddleWorld(slip_prob=0., max_steps=10)
            policy = rl.policy.ActionSequencePolicy([
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.right,
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.down
            ])
            rl.data.simulate(mdp, policy, logger)
        self.assertTrue(np.all(logger.get_episode_length() == np.ones(5) * 10))

    def _traj_equal(self, traj_1, traj_2):
        import numpy as np
        for i in [0, 1, 2, 3, 4]:
            self.assertTrue(np.allclose(traj_1.all()[i], traj_2.all()[i]))
        for d1, d2 in zip(traj_1.all()[5], traj_2.all()[5]):
            self.assertEqual(set(d1.keys()), set(d2.keys()))
            for k in d1.keys():
                self.assertEqual(d1[k], d2[k])

    def test_transition_listener(self):
        import rlutils as rl
        import numpy as np


        mdp = rl.environment.PuddleWorld(slip_prob=0., max_steps=10)
        buffer = rl.data.TrajectoryBuffer(mdp.transition_spec)
        logger = rl.logging.LoggerTrajectory(buffer)

        mdp = rl.environment.PuddleWorld(slip_prob=0.)
        action_seq = [
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.right,
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.down,
            rl.environment.gridworld.GridWorldAction.left
        ]
        policy = rl.policy.ActionSequencePolicy(action_seq)
        rl.data.simulate(mdp, policy, logger)

        self.assertEqual(buffer.num_transitions, 12)
        self.assertEqual(buffer.num_states, 13)
        
        r_seq = buffer.get_transition_column(rl.Reward)
        r_seq_corr = np.array(
            [0.,  0., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  1.], 
            dtype=np.float32
        )
        self.assertTrue(np.all(r_seq == r_seq_corr))

        a_seq = buffer.get_transition_column(rl.Action)
        self.assertTrue(np.all(a_seq == np.array(action_seq, dtype=np.int32)))

        x_seq = buffer.get_state_column(rl.environment.PuddleWorld.X)
        y_seq = buffer.get_state_column(rl.environment.PuddleWorld.Y)
        xy_seq = np.stack([x_seq, y_seq]).transpose()
        xy_seq_corr = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [1, 7],
            [1, 8],
            [1, 9],
            [1, 9],
            [0, 9]
        ], dtype=np.int32)
        self.assertTrue(np.all(xy_seq == xy_seq_corr))


    def test_transition_listener_timeout(self):
        import rlutils as rl
        import numpy as np

        mdp = rl.environment.PuddleWorld(slip_prob=0., max_steps=3)
        buffer = rl.data.TrajectoryBuffer(mdp.transition_spec)
        logger = rl.logging.LoggerTrajectory(buffer)

        mdp = rl.environment.PuddleWorld(slip_prob=0., max_steps=3)
        policy = rl.policy.ActionSequencePolicy(
            [rl.environment.gridworld.GridWorldAction.up] * 3
        )
        rl.data.simulate(mdp, policy, logger)
        self.assertEqual(buffer.num_transitions, 3)
        self.assertEqual(buffer.num_states, 4)
