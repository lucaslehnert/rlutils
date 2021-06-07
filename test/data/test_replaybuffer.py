from rlutils import data
from rlutils.data.replaybuffer import REWARD, TERM
from unittest import TestCase


# def _init_replay_buffer(max_num_transitions: int=None):
#     import numpy as np
#     import predictiverl as prl

#     state_cols = {
#         'img': np.zeros([100, 10, 10], dtype=np.int),
#         'c': np.zeros(100, dtype=np.int)
#     }
#     transition_cols = {
#         'action': np.zeros(100, dtype=np.int),
#         'reward': np.zeros(100, dtype=np.float)
#     }
#     buffer = prl.NumpyReplayBuffer(
#         state_cols, transition_cols, max_num_transitions=max_num_transitions
#     )
#     return buffer

def _init_trajectory():
    import numpy as np
    from rlutils.data.replaybuffer import Trajectory

    traj = Trajectory(
        state_defaults={
            'img': np.zeros([10, 10], dtype=np.int8),
            'c': np.int8(0)
        },
        transition_defaults={
            'action': np.int8(0),
            'reward': np.float32(0.)
        }
    )
    return traj


def _init_trajectory_10_step():
    import numpy as np

    traj = _init_trajectory()
    for i in range(10):
        traj.add_transition(
            state={
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
    return traj


def _init_trajectory_buffer():
    import numpy as np
    import rlutils as rl

    traj = rl.data.TrajectoryBuffer(
        state_defaults={
            'img': np.zeros([10, 10], dtype=np.int8),
            'c': np.int8(0)
        },
        transition_defaults={
            'action': np.int8(0),
            'reward': np.float32(0.)
        }
    )
    return traj


def _init_trajectory_buffer_10_step():
    import numpy as np

    buffer = _init_trajectory_buffer()
    for i in range(10):
        if i % 3 == 0 and i > 0:
            buffer.finish_current_sequence()
        buffer.add_transition(
            state={
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

    buffer = rl.data.TrajectoryBuffer(
        state_defaults={
            's': np.int32(0),
            'f': False
        },
        transition_defaults={
            rl.data.ACTION: 0,
            rl.data.REWARD: 0.,
            rl.data.TERM: False
        }
    )
    buffer.add_transition(
        {'s': 1, 'f': False},
        {rl.data.ACTION: 0, rl.data.REWARD: 0., rl.data.TERM: False},
        {'s': 2, 'f': False}
    )
    buffer.add_transition(
        {'s': 2, 'f': False},
        {rl.data.ACTION: 0, rl.data.REWARD: 0., rl.data.TERM: False},
        {'s': 3, 'f': False}
    )
    buffer.add_transition(
        {'s': 3, 'f': False},
        {rl.data.ACTION: 0, rl.data.REWARD: 0., rl.data.TERM: True},
        {'s': 4, 'f': True}
    )
    return buffer

class TestReplayBuffer(TestCase):
    def test_trajectory_init(self):
        traj = _init_trajectory()
        self.assertEqual(traj.num_states(), 0)
        self.assertEqual(traj.num_transitions(), 0)

    def test_trajectory_init_empty_state_columns(self):
        import rlutils as rl
        import numpy as np
        from rlutils.data.replaybuffer import Trajectory

        try:
            Trajectory(
                state_defaults={},
                transition_defaults={
                    'action': np.int8(0),
                    'reward': np.float32(0.)
                }
            )
            self.fail()
        except rl.data.ReplayBufferException:
            pass

    def test_trajectory_init_empty_transition_columns(self):
        import rlutils as rl
        import numpy as np
        from rlutils.data.replaybuffer import Trajectory

        try:
            Trajectory(
                state_defaults={
                    'img': np.zeros([10, 10], dtype=np.int8),
                    'c': np.int8(0)
                },
                transition_defaults={}
            )
            self.fail()
        except rl.data.ReplayBufferException:
            pass

    def test_trajectory_transition_column_names(self):
        traj = _init_trajectory()
        self.assertListEqual(
            traj.get_transition_column_names(), ['action', 'reward']
        )

    def test_trajectory_state_column_names(self):
        traj = _init_trajectory()
        self.assertListEqual(
            traj.get_state_column_names(), ['img', 'c']
        )

    def test_trajectory_add_transitions(self):
        import numpy as np

        traj = _init_trajectory_10_step()
        self.assertEqual(traj.num_transitions(), 10)
        self.assertEqual(traj.num_states(), 11)

        self.assertTrue(np.all(traj.get_column('action') == np.arange(10)))
        self.assertTrue(np.all(traj.get_column('reward') == np.ones(10) * 0.5))

        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i in range(11):
            img_corr[i] *= i
        self.assertTrue(np.all(traj.get_state_column('img') == img_corr))
        c_corr = np.ones(11, dtype=np.int8)
        self.assertTrue(np.all(traj.get_state_column('c') == c_corr))
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
        act_seq = traj.get_column('action', idxs=[4, 5, 8, 1])
        self.assertTrue(np.all(act_seq == np.array([4, 5, 8, 1])))

        act_seq = np.zeros(4)
        traj.set_column('action', act_seq, idxs=[4, 5, 8, 1])
        act_seq = traj.get_column('action', idxs=[4, 5, 8, 1])
        self.assertTrue(np.all(act_seq == 0))

    def test_trajectory_state_column(self):
        import numpy as np

        traj = _init_trajectory_10_step()
        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i in range(11):
            img_corr[i] *= i

        img = traj.get_state_column('img', idxs=[1, 2, 6, 10])
        self.assertTrue(np.all(img == img_corr[[1, 2, 6, 10]]))

        img = np.zeros([4, 10, 10], dtype=np.int8)
        traj.set_state_column('img', img, idxs=[1, 2, 6, 10])

        img = traj.get_state_column('img', idxs=[1, 2, 6, 10])
        self.assertTrue(np.all(img == 0))

    def test_trajectory_start_state_column(self):
        import numpy as np

        traj = _init_trajectory_10_step()
        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i in range(11):
            img_corr[i] *= i

        img = traj.get_start_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(img == img_corr[[1, 2, 6, 9]]))

        img = np.zeros([4, 10, 10], dtype=np.int8)
        traj.set_start_state_column('img', img, idxs=[1, 2, 6, 9])

        img = traj.get_start_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(img == 0))

    def test_trajectory_next_state_column(self):
        import numpy as np

        traj = _init_trajectory_10_step()
        img_corr = np.ones([11, 10, 10], dtype=np.int8)
        for i in range(11):
            img_corr[i] *= i

        img = traj.get_next_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(img == img_corr[[2, 3, 7, 10]]))

        img = np.zeros([4, 10, 10], dtype=np.int8)
        traj.set_next_state_column('img', img, idxs=[1, 2, 6, 9])

        img = traj.get_next_state_column('img', idxs=[1, 2, 6, 9])
        self.assertTrue(np.all(img == 0))

    def test_trajectory_buffer_init(self):
        buffer = _init_trajectory_buffer()
        self.assertEqual(buffer.num_states(), 0)
        self.assertEqual(buffer.num_transitions(), 0)

    def test_trajectory_buffer_add_transitions(self):
        import numpy as np

        buf = _init_trajectory_buffer_10_step()

        self.assertEqual(buf.num_states(), 14)
        self.assertEqual(buf.num_transitions(), 10)
        self.assertTrue(np.all(np.arange(10) == buf.get_column('action')))
        self.assertTrue(np.all(np.ones(10) * 0.5 == buf.get_column('reward')))

        img_corr = np.ones([14, 10, 10], dtype=np.int8)
        for i, v in enumerate([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10]):
            img_corr[i] *= v
        img = buf.get_state_column('img')
        self.assertTrue(np.all(img == img_corr))

        self.assertTrue(np.all(buf.get_state_column('c') == np.ones(14)))

    def test_trajectory_buffer_column(self):
        import numpy as np

        traj = _init_trajectory_buffer_10_step()
        act_seq = traj.get_column('action', idxs=[4, 5, 8, 1])
        self.assertTrue(np.all(act_seq == np.array([4, 5, 8, 1])))

        act_seq = np.zeros(4)
        traj.set_column('action', act_seq, idxs=[4, 5, 8, 1])
        act_seq = traj.get_column('action', idxs=[4, 5, 8, 1])
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

        buf = rl.data.TrajectoryBufferFixedSize(
            state_defaults={
                'img': np.zeros([10, 10], dtype=np.int8),
                'c': np.int8(0)
            },
            transition_defaults={
                'action': np.int8(0),
                'reward': np.float32(0.)
            },
            max_transitions=5
        )
        for i in range(10):
            if i % 3 == 0 and i > 0:
                buf.finish_current_sequence()
            buf.add_transition(
                state={
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

        self.assertEqual(buf.num_states(), 8)
        self.assertEqual(buf.num_transitions(), 5)

        act_corr = np.array([5, 6, 7, 8, 9])
        self.assertTrue(np.all(act_corr == buf.get_column('action')))
        self.assertTrue(np.all(np.ones(5) * 0.5 == buf.get_column('reward')))

        img_corr = np.ones([14, 10, 10], dtype=np.int8)
        for i, v in enumerate([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10]):
            img_corr[i] *= v
        img = buf.get_state_column('img')
        self.assertTrue(np.all(img == img_corr[-8:]))

        self.assertTrue(np.all(buf.get_state_column('c') == np.ones(8)))

    def test_h5_file_persistance(self):
        import rlutils as rl
        import numpy as np

        traj = _init_trajectory_buffer_10_step()
        traj.save_h5('test.h5')

        traj_recon = rl.data.TrajectoryBuffer.load_h5('test.h5')

        self.assertEqual(traj.num_states(), traj_recon.num_states())
        self.assertEqual(traj.num_transitions(), traj_recon.num_transitions())

        self.assertEqual(traj_recon.num_states(), 14)
        self.assertEqual(traj_recon.num_transitions(), 10)
        self.assertTrue(np.all(
            np.arange(10) == traj_recon.get_column('action')
        ))
        self.assertTrue(np.all(
            np.ones(10) * 0.5 == traj_recon.get_column('reward')
        ))

        img_corr = np.ones([14, 10, 10], dtype=np.int8)
        for i, v in enumerate([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10]):
            img_corr[i] *= v
        img = traj_recon.get_state_column('img')
        self.assertTrue(np.all(img == img_corr))

        self.assertTrue(
            np.all(traj_recon.get_state_column('c') == np.ones(14)))
        
        import os
        os.remove('test.h5')

    def test_add_transitions_transition_column_filter(self):
        import numpy as np
        import rlutils as rl

        buf = _init_trajectory_buffer_10_step()
        buf = buf.filter_transition_column(['action'])

        try:
            buf.get_column('reward')
            self.fail()
        except rl.data.ReplayBufferException:
            pass

        try:
            buf.set_column('reward', np.zeros(4), idxs=np.arange(4))
            self.fail()
        except rl.data.ReplayBufferException:
            pass

        self.assertEqual(buf.num_states(), 14)
        self.assertEqual(buf.num_transitions(), 10)
        self.assertTrue(np.all(np.arange(10) == buf.get_column('action')))

        img_corr = np.ones([14, 10, 10], dtype=np.int8)
        for i, v in enumerate([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10]):
            img_corr[i] *= v
        img = buf.get_state_column('img')
        self.assertTrue(np.all(img == img_corr))

        self.assertTrue(np.all(buf.get_state_column('c') == np.ones(14)))

    def test_add_transitions_state_column_filter(self):
        import numpy as np
        import rlutils as rl

        buf = _init_trajectory_buffer_10_step()
        buf = buf.filter_state_column(['img'])

        self.assertEqual(buf.num_states(), 14)
        self.assertEqual(buf.num_transitions(), 10)
        self.assertTrue(np.all(np.arange(10) == buf.get_column('action')))
        self.assertTrue(np.all(np.ones(10) * 0.5 == buf.get_column('reward')))

        img_corr = np.ones([14, 10, 10], dtype=np.int8)
        for i, v in enumerate([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10]):
            img_corr[i] *= v
        img = buf.get_state_column('img')
        self.assertTrue(np.all(img == img_corr))

        try:
            buf.get_state_column('c')
            self.fail()
        except rl.data.ReplayBufferException:
            pass
        try:
            buf.get_start_state_column('c')
            self.fail()
        except rl.data.ReplayBufferException:
            pass
        try:
            buf.get_next_state_column('c')
            self.fail()
        except rl.data.ReplayBufferException:
            pass

        try:
            buf.set_state_column('c', np.zeros(4), np.arange(4))
            self.fail()
        except rl.data.ReplayBufferException:
            pass
        try:
            buf.set_start_state_column('c', np.zeros(4), np.arange(4))
            self.fail()
        except rl.data.ReplayBufferException:
            pass
        try:
            buf.set_next_state_column('c', np.zeros(4), np.arange(4))
            self.fail()
        except rl.data.ReplayBufferException:
            pass

    def test_transition_iterator_epochs(self):
        import rlutils as rl
        import numpy as np

        buffer = _three_transition_buffer()
        t_it = rl.data.TransitionIteratorEpochs(
            buffer, batch_size=2, shuffle=False
        )
        s_l = []
        f_l = []
        a_l = []
        r_l = []
        t_l = []
        sn_l = []
        fn_l = []
        for s, f, a, r, t, sn, fn in t_it:
            s_l.append(s)
            f_l.append(f)
            a_l.append(a)
            r_l.append(r)
            t_l.append(t)
            sn_l.append(sn)
            fn_l.append(fn)
        s_l = np.concatenate(s_l)
        f_l = np.concatenate(f_l)
        a_l = np.concatenate(a_l)
        r_l = np.concatenate(r_l)
        t_l = np.concatenate(t_l)
        sn_l = np.concatenate(sn_l)
        fn_l = np.concatenate(fn_l)
        self.assertTrue(np.all(s_l == np.array([1, 2, 3])))
        self.assertTrue(np.all(f_l == np.array([False, False, False])))
        self.assertTrue(np.all(a_l == np.array([0, 0, 0])))
        self.assertTrue(np.all(r_l == np.array([0., 0., 0.])))
        self.assertTrue(np.all(t_l == np.array([False, False, True])))
        self.assertTrue(np.all(sn_l == np.array([2, 3, 4])))
        self.assertTrue(np.all(fn_l == np.array([False, False, True])))

    def test_transition_iterator_sampled(self):
        import rlutils as rl
        import numpy as np

        buffer = _three_transition_buffer()
        t_it = rl.data.TransitionIteratorSampled(
            buffer, batch_size=2
        )
        for i, (s, f, a, r, t, sn, fn) in enumerate(t_it):            
            self.assertEqual(np.shape(s)[0], 2)
            self.assertEqual(np.shape(f)[0], 2)
            self.assertEqual(np.shape(a)[0], 2)
            self.assertEqual(np.shape(r)[0], 2)
            self.assertEqual(np.shape(t)[0], 2)
            self.assertEqual(np.shape(sn)[0], 2)
            self.assertEqual(np.shape(fn)[0], 2)
        self.assertEqual(i, 999)

    def test_state_iterator_epochs(self):
        import rlutils as rl
        import numpy as np

        buffer = _three_transition_buffer()
        t_it = rl.data.StateIteratorEpochs(
            buffer, batch_size=2, shuffle=False
        )
        s_l = []
        f_l = []
        for s, f in t_it:
            s_l.append(s)
            f_l.append(f)
        s_l = np.concatenate(s_l)
        f_l = np.concatenate(f_l)
        self.assertTrue(np.all(
            s_l == np.array([1, 2, 3, 4])
        ))
        self.assertTrue(np.all(
            f_l == np.array([False, False, False, True])
        ))
    
    def test_state_iterator_sampled(self):
        import rlutils as rl
        import numpy as np

        buffer = _three_transition_buffer()
        t_it = rl.data.StateIteratorSampled(
            buffer, batch_size=2
        )
        for i, (s, f) in enumerate(t_it):
            self.assertEqual(np.shape(s)[0], 2)
            self.assertEqual(np.shape(f)[0], 2)
        self.assertEqual(i, 999)
