#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from unittest import TestCase


def _transition_in_list(transition_list, s, a, r, sn, t=None, info=None):
    import numpy as np
    for sc, ac, rc, snc, tc, infoc in transition_list:
        same_s = np.allclose(s, sc)
        same_a = (a == ac)
        same_r = (r == rc)
        same_sn = np.allclose(sn, snc)
        if t is not None:
            same_t = (t == tc)
        else:
            same_t = True
        if info is not None:
            same_info = (info == infoc)
        else:
            same_info = True
        if same_s and same_a and same_r and same_sn and same_t and same_info:
            return True
    return False


def _transition_batch_in_list(transition_list, sb, ab, rb, snb, tb=None, infob=None):
    import numpy as np
    if tb is None:
        tb = [None] * len(ab)
    if infob is None:
        infob = [None] * len(ab)
    it = zip(sb, ab, rb, snb, tb, infob)
    return np.all([_transition_in_list(transition_list, s, a, r, sn, t, i) for s, a, r, sn, t, i in it])


class TestBufferSamplerUniform(TestCase):
    def test_buffer_sampler_uniform(self):
        import rlutils as rl
        import numpy as np
        buffer = rl.data.TransitionBuffer()
        transition_list = [
            ([0], 0, 0., [1], False, {}),
            ([1], 0, 0., [0], False, {}),
            ([1], 0, 1., [1], True, {})
        ]
        for t in transition_list:
            buffer.update_transition(*t)

        for s, a, r, sn, t, i in rl.data.BufferSamplerUniform(buffer, batch_size=2, num_samples=10):
            self.assertEqual(np.shape(s), (2, 1))
            self.assertEqual(np.shape(a), (2,))
            self.assertEqual(np.shape(r), (2,))
            self.assertEqual(np.shape(sn), (2, 1))
            self.assertEqual(np.shape(t), (2,))

            self.assertTrue(_transition_batch_in_list(transition_list, s, a, r, sn, t, i))

    def test_buffer_sampler_uniform_SARST(self):
        import rlutils as rl
        import numpy as np
        buffer = rl.data.TransitionBuffer()
        transition_list = [
            ([0], 0, 0., [1], False, {}),
            ([1], 0, 0., [0], False, {}),
            ([1], 0, 1., [1], True, {})
        ]
        for t in transition_list:
            buffer.update_transition(*t)

        for s, a, r, sn, t in rl.data.BufferSamplerUniformSARST(buffer, batch_size=2, num_samples=10):
            self.assertEqual(np.shape(s), (2, 1))
            self.assertEqual(np.shape(a), (2,))
            self.assertEqual(np.shape(r), (2,))
            self.assertEqual(np.shape(sn), (2, 1))
            self.assertEqual(np.shape(t), (2,))

            self.assertTrue(_transition_batch_in_list(transition_list, s, a, r, sn, t))

    def test_buffer_sampler_uniform_SARS(self):
        import rlutils as rl
        import numpy as np
        buffer = rl.data.TransitionBuffer()
        transition_list = [
            ([0], 0, 0., [1], False, {}),
            ([1], 0, 0., [0], False, {}),
            ([1], 0, 1., [1], True, {})
        ]
        for t in transition_list:
            buffer.update_transition(*t)

        for s, a, r, sn in rl.data.BufferSamplerUniformSARS(buffer, batch_size=2, num_samples=10):
            self.assertEqual(np.shape(s), (2, 1))
            self.assertEqual(np.shape(a), (2,))
            self.assertEqual(np.shape(r), (2,))
            self.assertEqual(np.shape(sn), (2, 1))

            self.assertTrue(_transition_batch_in_list(transition_list, s, a, r, sn))

    def test_buffer_uniform_sampler_exception(self):
        import rlutils as rl
        buffer = rl.data.TransitionBuffer()
        transition_list = [
            ([0], 0, 0., [1], False, {}),
            ([1], 0, 0., [0], False, {}),
            ([1], 0, 1., [1], True, {})
        ]
        for t in transition_list:
            buffer.update_transition(*t)

        try:
            for _ in rl.data.BufferSamplerUniform(buffer, batch_size=4, num_samples=10):
                self.fail()
        except rl.data.BufferIteratorException:
            pass

    def test_buffer_iterator_shuffle(self):
        import rlutils as rl
        import numpy as np
        buffer = rl.data.TransitionBuffer()
        transition_list = [
            ([0], 0, 0., [1], False, {}),
            ([1], 0, 0., [0], False, {}),
            ([1], 0, 1., [1], True, {})
        ]
        for t in transition_list:
            buffer.update_transition(*t)

        for s, a, r, sn, t, i in rl.data.BufferIterator(buffer, batch_size=2, num_samples=10):
            self.assertEqual(np.shape(s), (2, 1))
            self.assertEqual(np.shape(a), (2,))
            self.assertEqual(np.shape(r), (2,))
            self.assertEqual(np.shape(sn), (2, 1))
            self.assertEqual(np.shape(t), (2,))

            self.assertTrue(_transition_batch_in_list(transition_list, s, a, r, sn, t, i))

    def test_buffer_iterator(self):
        import rlutils as rl
        import numpy as np
        buffer = rl.data.TransitionBuffer()
        transition_list = [
            ([0], 0, 0., [1], False, {}),
            ([1], 1, 0., [0], False, {}),
            ([1], 0, 1., [1], True, {})
        ]
        for t in transition_list:
            buffer.update_transition(*t)

        for i, (s, a, r, sn, t, info) in enumerate(rl.data.BufferIterator(buffer, 2, 2, shuffle=False)):
            self.assertEqual(np.shape(s), (2, 1))
            self.assertEqual(np.shape(a), (2,))
            self.assertEqual(np.shape(r), (2,))
            self.assertEqual(np.shape(sn), (2, 1))
            self.assertEqual(np.shape(t), (2,))

            if i == 0:
                self.assertTrue(np.allclose(s, [[0], [1]]))
                self.assertTrue(np.all(a == [0, 1]))
                self.assertTrue(np.all(r == [0., 0.]))
                self.assertTrue(np.allclose(sn, [[1], [0]]))
                self.assertTrue(np.all(t == [False, False]))
                self.assertTrue(np.all([len(i) == 0 for i in info]))
            elif i == 1:
                self.assertTrue(np.allclose(s, [[1], [0]]))
                self.assertTrue(np.all(a == [0, 0]))
                self.assertTrue(np.all(r == [1., 0.]))
                self.assertTrue(np.allclose(sn, [[1], [1]]))
                self.assertTrue(np.all(t == [True, False]))
                self.assertTrue(np.all([len(i) == 0 for i in info]))

    def test_buffer_iterator_SARST(self):
        import rlutils as rl
        import numpy as np
        buffer = rl.data.TransitionBuffer()
        transition_list = [
            ([0], 0, 0., [1], False, {}),
            ([1], 1, 0., [0], False, {}),
            ([1], 0, 1., [1], True, {})
        ]
        for t in transition_list:
            buffer.update_transition(*t)

        for i, (s, a, r, sn, t) in enumerate(rl.data.BufferIteratorSARST(buffer, 2, 2, shuffle=False)):
            self.assertEqual(np.shape(s), (2, 1))
            self.assertEqual(np.shape(a), (2,))
            self.assertEqual(np.shape(r), (2,))
            self.assertEqual(np.shape(sn), (2, 1))
            self.assertEqual(np.shape(t), (2,))

            if i == 0:
                self.assertTrue(np.allclose(s, [[0], [1]]))
                self.assertTrue(np.all(a == [0, 1]))
                self.assertTrue(np.all(r == [0., 0.]))
                self.assertTrue(np.allclose(sn, [[1], [0]]))
                self.assertTrue(np.all(t == [False, False]))
            elif i == 1:
                self.assertTrue(np.allclose(s, [[1], [0]]))
                self.assertTrue(np.all(a == [0, 0]))
                self.assertTrue(np.all(r == [1., 0.]))
                self.assertTrue(np.allclose(sn, [[1], [1]]))
                self.assertTrue(np.all(t == [True, False]))

    def test_buffer_iterator_SARS(self):
        import rlutils as rl
        import numpy as np
        buffer = rl.data.TransitionBuffer()
        transition_list = [
            ([0], 0, 0., [1], False, {}),
            ([1], 1, 0., [0], False, {}),
            ([1], 0, 1., [1], True, {})
        ]
        for t in transition_list:
            buffer.update_transition(*t)

        for i, (s, a, r, sn) in enumerate(rl.data.BufferIteratorSARS(buffer, 2, 2, shuffle=False)):
            self.assertEqual(np.shape(s), (2, 1))
            self.assertEqual(np.shape(a), (2,))
            self.assertEqual(np.shape(r), (2,))
            self.assertEqual(np.shape(sn), (2, 1))

            if i == 0:
                self.assertTrue(np.allclose(s, [[0], [1]]))
                self.assertTrue(np.all(a == [0, 1]))
                self.assertTrue(np.all(r == [0., 0.]))
                self.assertTrue(np.allclose(sn, [[1], [0]]))
            elif i == 1:
                self.assertTrue(np.allclose(s, [[1], [0]]))
                self.assertTrue(np.all(a == [0, 0]))
                self.assertTrue(np.all(r == [1., 0.]))
                self.assertTrue(np.allclose(sn, [[1], [1]]))
