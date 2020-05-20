#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestLoggerTotalReward(TestCase):
    def test_total_reward(self):
        import rlutils as rl
        import numpy as np
        logger = rl.logging.LoggerTotalReward()
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
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.DOWN,
                rl.environment.gridworld.GridWorldAction.LEFT
            ])
            rl.data.simulate(mdp, policy, logger)
        self.assertTrue(np.all(logger.get_total_reward_episodic() == np.ones(5) * -6.))

    def test_finish_trajectory(self):
        import rlutils as rl
        import numpy as np
        logger = rl.logging.LoggerTotalReward()
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
            try:
                rl.data.simulate(mdp, policy, logger, max_steps=10)
            except rl.data.SimulationTimout:
                logger.finish_episode()
        self.assertTrue(np.all(logger.get_total_reward_episodic() == np.ones(5) * -7.))


