#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestSimulate(TestCase):
    def test_simulate_gracefully(self):
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
            rl.data.simulate_gracefully(mdp, policy, logger, max_steps=10)
            logger.update_episode_length(10)
        self.assertTrue(np.all(logger.get_episode_length() == np.ones(5) * 10))

    def _traj_equal(self, traj_1, traj_2):
        import numpy as np
        for i in [0, 1, 2, 3, 4]:
            self.assertTrue(np.allclose(traj_1.all()[i], traj_2.all()[i]))
        for d1, d2 in zip(traj_1.all()[5], traj_2.all()[5]):
            self.assertEqual(set(d1.keys()), set(d2.keys()))
            for k in d1.keys():
                self.assertEqual(d1[k], d2[k])

    def test_replay_trajectory(self):
        import rlutils as rl
        logger = rl.logging.LoggerTrajectory()

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
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.LEFT
        ])
        rl.data.simulate(mdp, policy, logger)
        traj_1 = logger.get_trajectory_list()[0]

        logger = rl.logging.LoggerTrajectory()
        rl.data.replay_trajectory(traj_1, logger)
        traj_2 = logger.get_trajectory_list()[0]

        self._traj_equal(traj_1, traj_2)

    def test_transition_listener(self):
        import rlutils as rl
        logger_1 = rl.logging.LoggerTrajectory()
        logger_2 = rl.logging.LoggerTrajectory()

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
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.DOWN,
            rl.environment.gridworld.GridWorldAction.LEFT
        ])
        rl.data.simulate(mdp, policy, rl.data.transition_listener(logger_1, logger_2))
        traj_1 = logger_1.get_trajectory_list()[0]
        traj_2 = logger_2.get_trajectory_list()[0]

        self._traj_equal(traj_1, traj_2)
