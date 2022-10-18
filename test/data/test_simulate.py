#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from rlutils.environment.TabularMDP import TabularMDP
from rlutils.data.replaybuffer import Reward, Term
from rlutils.environment.PuddleWorld import PuddleWorld
from rlutils import data, environment
from unittest import TestCase


class TestSimulate(TestCase):
    def test_simulate(self):
        import rlutils as rl
        import numpy as np
        logger = rl.logging.LoggerEpisodeLength()
        for _ in range(5):
            mdp = rl.environment.PuddleWorld(slip_prob=0.)
            policy = rl.policy.ActionSequencePolicy([
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.RIGHT,
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.DOWN
            ])
            rl.data.simulate(mdp, policy, logger, max_steps=10)
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


        mdp = rl.environment.PuddleWorld(slip_prob=0.)
        zero_vec = np.zeros(100, dtype=np.float32)
        buffer = rl.data.TrajectoryBuffer(
            state_defaults={
                rl.environment.TabularMDP.ONE_HOT: zero_vec,
                rl.environment.PuddleWorld.X: np.int32(0),
                rl.environment.PuddleWorld.Y: np.int32(0)
            },
            transition_defaults={
                rl.data.Action: np.int32(0),
                rl.data.Reward: np.float32(0.),
                rl.data.Term: False
            }
        )
        logger = rl.logging.LoggerTrajectory(buffer)

        mdp = rl.environment.PuddleWorld(slip_prob=0.)
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
            rl.environment.gridworld.GridWorldAction.LEFT
        ]
        policy = rl.policy.ActionSequencePolicy(action_seq)
        rl.data.simulate(mdp, policy, logger)

        self.assertEqual(buffer.num_transitions(), 12)
        self.assertEqual(buffer.num_states(), 13)
        
        r_seq = buffer.get_column(rl.data.Reward)
        r_seq_corr = np.array(
            [0.,  0., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  1.], 
            dtype=np.float32
        )
        self.assertTrue(np.all(r_seq == r_seq_corr))

        a_seq = buffer.get_column(rl.data.Action)
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

        mdp = rl.environment.PuddleWorld(slip_prob=0.)
        zero_vec = np.zeros(100, dtype=np.float32)
        buffer = rl.data.TrajectoryBuffer(
            state_defaults={
                rl.environment.TabularMDP.ONE_HOT: zero_vec,
                rl.environment.PuddleWorld.X: np.int32(0),
                rl.environment.PuddleWorld.Y: np.int32(0)
            },
            transition_defaults={
                rl.data.Action: np.int32(0),
                rl.data.Reward: np.float32(0.),
                rl.data.Term: False
            }
        )
        logger = rl.logging.LoggerTrajectory(buffer)

        mdp = rl.environment.PuddleWorld(slip_prob=0.)
        policy = rl.policy.ActionSequencePolicy(
            [rl.environment.gridworld.GridWorldAction.UP] * 101
        )
        rl.data.simulate(mdp, policy, logger, max_steps=100)
        self.assertEqual(buffer.num_transitions(), 100)
        self.assertEqual(buffer.num_states(), 101)
