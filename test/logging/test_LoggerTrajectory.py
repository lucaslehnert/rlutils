#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


class TestLoggerTotalReward(TestCase):
    def test_trajectory(self):
        import rlutils as rl
        import numpy as np
        logger = rl.logging.LoggerTrajectory()

        act_seq = [
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
            (0, 9)
        ]
        s_seq_corr = [rl.environment.gridworld.pt_to_idx(xy, (10, 10)) for xy in xy_seq]

        for _ in range(5):
            mdp = rl.environment.PuddleWorld(slip_prob=0.)
            policy = rl.policy.ActionSequencePolicy(act_seq)
            rl.data.simulate(mdp, policy, logger)

        for traj in logger.get_trajectory_list():
            s, a, r, sn, t, _ = traj.all()
            self.assertTrue(np.all(np.array([0., 0., -1., -1., -1., -1., -1., -1., -1., 0., 0., 1.]) == r))
            self.assertTrue(np.all(np.where(s == 1.)[1] == s_seq_corr[:-1]))
            self.assertTrue(np.all(np.where(sn == 1.)[1] == s_seq_corr[1:]))
            self.assertTrue(np.all(a == act_seq))
            self.assertTrue(np.all(t == [False] * 11 + [True]))

    def test_finish_episode(self):
        def test_trajectory(self):
            import rlutils as rl
            import numpy as np
            logger = rl.logging.LoggerTrajectory()

            act_seq = [
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
                (1, 9)
            ]
            s_seq_corr = [rl.environment.gridworld.pt_to_idx(xy, (10, 10)) for xy in xy_seq]

            for _ in range(5):
                mdp = rl.environment.PuddleWorld(slip_prob=0.)
                policy = rl.policy.ActionSequencePolicy(act_seq)
                try:
                    rl.data.simulate(mdp, policy, logger, max_steps=10)
                except rl.data.SimulationTimeout:
                    logger.on_simulation_timeout()

            for traj in logger.get_trajectory_list():
                s, a, r, sn, t, _ = traj.all()
                self.assertTrue(np.all(np.array([0., 0., -1., -1., -1., -1., -1., -1., -1., 0., 0.]) == r))
                self.assertTrue(np.all(np.where(s == 1.)[1] == s_seq_corr[:-1]))
                self.assertTrue(np.all(np.where(sn == 1.)[1] == s_seq_corr[1:]))
                self.assertTrue(np.all(a == act_seq))
                self.assertTrue(np.all(t == [False] * 11))
