#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestLoggerEpisodeLength(TestCase):
    def test_episode_length(self):
        import rlutils as rl
        import numpy as np
        logger = rl.logging.LoggerEpisodeLength()
        for _ in range(5):
            mdp = rl.environment.PuddleWorld(slip_prob=0.)
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
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.down,
                rl.environment.gridworld.GridWorldAction.left
            ])
            rl.data.simulate(mdp, policy, logger)
        self.assertTrue(np.all(logger.get_episode_length() == np.ones(5) * 12))

    def test_finish_trajectory(self):
        import rlutils as rl
        import numpy as np
        logger = rl.logging.LoggerEpisodeLength()
        for _ in range(5):
            mdp = rl.environment.PuddleWorld(slip_prob=0.)
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
            try:
                rl.data.simulate(mdp, policy, logger, max_steps=10)
            except rl.data.SimulationTimeout:
                logger.on_simulation_timeout()
        self.assertTrue(np.all(logger.get_episode_length() == np.ones(5) * 10))

    def test_update_episode_length(self):
        import rlutils as rl
        import numpy as np
        logger = rl.logging.LoggerEpisodeLength()
        for _ in range(5):
            mdp = rl.environment.PuddleWorld(slip_prob=0.)
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
            try:
                rl.data.simulate(mdp, policy, logger, max_steps=10)
            except rl.data.SimulationTimeout:
                logger.update_episode_length(10)
        self.assertTrue(np.all(logger.get_episode_length() == np.ones(5) * 10))
