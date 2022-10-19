from unittest import TestCase


def _init_trajectory():
    import numpy as np
    import rlutils as rl

    traj = rl.data.Trajectory()
    traj.add_state_column(rl.Column(
        name="img", shape=(10, 10), dtype=np.uint8
    ))
    traj.add_state_column(rl.Column(
        name="c", shape=(), dtype=np.int32
    ))
    traj.add_transition_column(rl.Column(
        name="action", shape=(), dtype=np.int32
    ))
    traj.add_transition_column(rl.Column(
        name="reward", shape=(), dtype=np.float32
    ))
    return traj


def _init_trajectory_10_step():
    import numpy as np

    traj = _init_trajectory()
    for i in range(10):
        traj.add_transition(
            start_state={
                'img': np.ones([10, 10], dtype=np.uint8) * i,
                'c': np.int32(1)
            },
            transition={
                'action': np.int32(i),
                'reward': np.float32(0.5)
            },
            next_state={
                'img': np.ones([10, 10], dtype=np.uint8) * (i + 1),
                'c': np.int32(1)
            }
        )
    return traj


def _init_trajectory_buffer():
    import numpy as np
    import rlutils as rl

    buffer = rl.data.TrajectoryBuffer()
    buffer.add_state_column(
        rl.Column(name="img", shape=(10, 10), dtype=np.uint8))
    buffer.add_state_column(
        rl.Column(name="c", shape=(), dtype=np.int32))
    buffer.add_transition_column(
        rl.Column(name="action", shape=(), dtype=np.int32))
    buffer.add_transition_column(
        rl.Column(name="reward", shape=(), dtype=np.float32))
    return buffer


def _init_trajectory_buffer_10_step():
    import numpy as np

    buffer = _init_trajectory_buffer()
    for i in range(10):
        if i % 3 == 0 and i > 0:
            buffer.finish_current_sequence()
        buffer.add_transition(
            start_state={
                'img': np.ones([10, 10], dtype=np.int8) * i,
                'c': 1
            },
            transition={
                'action': i,
                'reward': 0.5
            },
            next_state={
                'img': np.ones([10, 10], dtype=np.int8) * (i + 1),
                'c': 1
            }
        )
    return buffer


def _three_transition_buffer():
    import rlutils as rl
    import numpy as np

    buffer = rl.data.TrajectoryBuffer()
    buffer.add_state_column(rl.Column(name="s", shape=(), dtype=np.int32))
    buffer.add_state_column(rl.Column(name="f", shape=(), dtype=np.bool))
    buffer.add_transition_column(rl.Action_index_column)
    buffer.add_transition_column(rl.Reward_column)
    buffer.add_transition_column(rl.Term_column)
    buffer.add_transition(
        {'s': np.int32(1), 'f': False},
        {
            rl.Action: np.int32(0),
            rl.Reward: np.float32(0.),
            rl.Term: False
        },
        {'s': np.int32(2), 'f': False}
    )
    buffer.add_transition(
        {'s': np.int32(2), 'f': False},
        {
            rl.Action: np.int32(0),
            rl.Reward: np.float32(0.),
            rl.Term: False
        },
        {'s': np.int32(3), 'f': False}
    )
    buffer.add_transition(
        {'s': np.int32(3), 'f': False},
        {
            rl.Action: np.int32(0),
            rl.Reward: np.float32(0.),
            rl.Term: True
        },
        {'s': np.int32(4), 'f': True}
    )
    return buffer


class TestReplayBuffer(TestCase):
    def test_trajectory_init(self):
        traj = _init_trajectory()
        self.assertEqual(traj.num_states, 0)
        self.assertEqual(traj.num_transitions, 0)

    def test_trajectory_transition_column_names(self):
        traj = _init_trajectory()
        self.assertListEqual(
            list(traj.transition_columns.keys()), ['action', 'reward']
        )

    def test_trajectory_state_column_names(self):
        traj = _init_trajectory()
        self.assertListEqual(
            list(traj.state_columns.keys()), ['img', 'c']
        )

    def test_trajectory_add_transitions(self):
        import numpy as np

        traj = _init_trajectory_10_step()
        self.assertEqual(traj.num_transitions, 10)
        self.assertEqual(traj.num_states, 11)

        self.assertTrue(np.all(
            traj.get_transition_column('action') == np.arange(10)
        ))
        self.assertTrue(np.all(
            traj.get_transition_column('reward') == np.ones(10) * 0.5
        ))

        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i in range(11):
            img_corr[i] *= i
        self.assertTrue(np.all(
            traj.get_state_column('img') == img_corr
        ))
        c_corr = np.ones(11, dtype=np.int8)
        self.assertTrue(np.all(
            traj.get_state_column('c') == c_corr
        ))
        self.assertTrue(np.all(
            traj.get_start_state_column('img') == img_corr[:-1]
        ))
        self.assertTrue(np.all(
            traj.get_start_state_column('c') == c_corr[:-1]
        ))
        self.assertTrue(np.all(
            traj.get_next_state_column('img') == img_corr[1:]
        ))
        self.assertTrue(np.all(
            traj.get_next_state_column('c') == c_corr[1:]
        ))

    def test_trajectory_column(self):
        import numpy as np

        traj = _init_trajectory_10_step()
        act_seq = traj.get_transition_column('action', idxs=[4, 5, 8, 1])
        self.assertTrue(np.all(np.array(act_seq) == np.array([4, 5, 8, 1])))

        act_seq = np.zeros(4)
        traj.set_transition_column('action', act_seq, idxs=[4, 5, 8, 1])
        act_seq = traj.get_transition_column('action', idxs=[4, 5, 8, 1])
        self.assertTrue(np.all(np.array(act_seq) == 0))

    def test_trajectory_state_column(self):
        import numpy as np

        traj = _init_trajectory_10_step()
        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i in range(11):
            img_corr[i] *= i

        img = traj.get_state_column('img', idxs=[1, 2, 6, 10])
        self.assertTrue(np.all(np.array(img) == img_corr[[1, 2, 6, 10]]))

        img = np.zeros([4, 10, 10], dtype=np.int8)
        traj.set_state_column('img', img, idxs=[1, 2, 6, 10])

        img = traj.get_state_column('img', idxs=[1, 2, 6, 10])
        self.assertTrue(np.all(np.array(img) == 0))

    def test_trajectory_start_state_column(self):
        import numpy as np

        traj = _init_trajectory_10_step()
        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i in range(11):
            img_corr[i] *= i

        img = traj.get_start_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(np.array(img) == img_corr[[1, 2, 6, 9]]))

        img = np.zeros([4, 10, 10], dtype=np.int8)
        traj.set_start_state_column('img', img, idxs=[1, 2, 6, 9])

        img = traj.get_start_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(np.array(img) == 0))

    def test_trajectory_next_state_column(self):
        import numpy as np

        traj = _init_trajectory_10_step()
        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i in range(11):
            img_corr[i] *= i

        img = traj.get_next_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(np.array(img) == img_corr[[2, 3, 7, 10]]))

        img = np.zeros([4, 10, 10], dtype=np.int8)
        traj.set_next_state_column('img', img, idxs=[1, 2, 6, 9])

        img = traj.get_next_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(np.array(img) == 0))

    def test_trajectory_buffer_init(self):
        buffer = _init_trajectory_buffer()
        self.assertEqual(buffer.num_states, 0)
        self.assertEqual(buffer.num_transitions, 0)

    def test_trajectory_buffer_add_transitions(self):
        import numpy as np

        buf = _init_trajectory_buffer_10_step()

        self.assertEqual(buf.num_states, 14)
        self.assertEqual(buf.num_transitions, 10)
        self.assertTrue(np.all(
            buf.get_transition_column('action') == np.arange(10)
        ))
        self.assertTrue(np.all(
            buf.get_transition_column('reward') == np.ones(10) * 0.5
        ))

        img_corr = np.ones([14, 10, 10], dtype=np.int8)
        for i, v in enumerate([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10]):
            img_corr[i] *= v
        img = buf.get_state_column('img')
        self.assertTrue(np.all(img == img_corr))
        self.assertTrue(np.all(buf.get_state_column('c') == np.ones(14)))

    def test_trajectory_buffer_column(self):
        import numpy as np

        traj = _init_trajectory_buffer_10_step()
        act_seq = traj.get_transition_column('action', idxs=[4, 5, 8, 1])
        self.assertTrue(np.all(act_seq == np.array([4, 5, 8, 1])))

        act_seq = np.zeros(4)
        traj.set_transition_column('action', act_seq, idxs=[4, 5, 8, 1])
        act_seq = traj.get_transition_column('action', idxs=[4, 5, 8, 1])
        self.assertTrue(np.all(act_seq == 0))

    def test_trajectory_buffer_state_column(self):
        import numpy as np

        traj = _init_trajectory_buffer_10_step()
        img_corr = np.ones([14, 10, 10], dtype=np.int8)
        for i, v in enumerate([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10]):
            img_corr[i] *= v

        img = traj.get_state_column('img', idxs=[1, 2, 6, 10])
        self.assertTrue(np.all(img == img_corr[[1, 2, 6, 10]]))

        img = np.zeros([4, 10, 10], dtype=np.int8)
        traj.set_state_column('img', img, idxs=[1, 2, 6, 10])

        img = traj.get_state_column('img', idxs=[1, 2, 6, 10])
        self.assertTrue(np.all(img == 0))

    def test_trajectory_buffer_start_state_column(self):
        import numpy as np

        traj = _init_trajectory_buffer_10_step()
        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            img_corr[i] *= v

        img = traj.get_start_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(img == img_corr[[1, 2, 6, 9]]))

        img = np.zeros([4, 10, 10], dtype=np.int8)
        traj.set_start_state_column('img', img, idxs=[1, 2, 6, 9])

        img = traj.get_start_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(img == 0))

    def test_trajectory_buffer_next_state_column(self):
        import numpy as np

        traj = _init_trajectory_buffer_10_step()
        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i, v in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            img_corr[i] *= v

        img = traj.get_next_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(img == img_corr[[1, 2, 6, 9]]))

        img = np.zeros([4, 10, 10], dtype=np.int8)
        traj.set_next_state_column('img', img, idxs=[1, 2, 6, 9])

        img = traj.get_next_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(img == 0))

    def test_trajectory_buffer_length_constrainted(self):
        import numpy as np
        import rlutils as rl

        buffer = rl.data.TrajectoryBufferFixedSize(5)
        buffer.add_state_column(rl.Column(
            name="img", shape=(10, 10), dtype=np.uint8
        ))
        buffer.add_state_column(rl.Column(
            name="c", shape=(), dtype=np.uint8
        ))
        buffer.add_transition_column(rl.Column(
            name="action", shape=(), dtype=np.int8
        ))
        buffer.add_transition_column(rl.Column(
            name="reward", shape=(), dtype=np.float32
        ))
        for i in range(10):
            if i % 3 == 0 and i > 0:
                buffer.finish_current_sequence()
            buffer.add_transition(
                start_state={
                    'img': np.ones([10, 10], dtype=np.int8) * i,
                    'c': 1
                },
                transition={
                    'action': i,
                    'reward': 0.5
                },
                next_state={
                    'img': np.ones([10, 10], dtype=np.int8) * (i + 1),
                    'c': 1
                }
            )

        self.assertEqual(buffer.num_states, 8)
        self.assertEqual(buffer.num_transitions, 5)

        act_corr = np.array([5, 6, 7, 8, 9])
        self.assertTrue(np.all(
            act_corr == buffer.get_transition_column('action')
        ))
        self.assertTrue(np.all(
            np.ones(5) * 0.5 == buffer.get_transition_column('reward')
        ))

        img_corr = np.ones([14, 10, 10], dtype=np.int8)
        for i, v in enumerate([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10]):
            img_corr[i] *= v
        img = buffer.get_state_column('img')
        self.assertTrue(np.all(img == img_corr[-8:]))

        self.assertTrue(np.all(buffer.get_state_column('c') == np.ones(8)))
