#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import unittest


class TestPersistenceTrajectory(unittest.TestCase):
    def test_transition_buffer_fixed_size(self):
        import numpy as np
        import shutil
        import rlutils

        env = rlutils.environment.PuddleWorld()
        policy = rlutils.policy.uniform_random(4)
        buffer = rlutils.data.TransitionBufferFixedSize(10000)
        rlutils.data.simulate_gracefully(env, policy, buffer)
        self.assertFalse(buffer.full())
        while not buffer.full():
            rlutils.data.simulate_gracefully(env, policy, buffer)
        self.assertTrue(buffer.full())

        dir_name = 'test_dir'
        buffer.save(dir_name)
        buffer_reconstruct = rlutils.data.TransitionBuffer()
        buffer_reconstruct.load(dir_name)

        self.assertTrue(np.allclose(buffer.all()[0], buffer_reconstruct.all()[0]))
        self.assertTrue(np.allclose(buffer.all()[1], buffer_reconstruct.all()[1]))
        self.assertTrue(np.allclose(buffer.all()[2], buffer_reconstruct.all()[2]))
        self.assertTrue(np.allclose(buffer.all()[3], buffer_reconstruct.all()[3]))
        self.assertTrue(np.allclose(buffer.all()[4], buffer_reconstruct.all()[4]))

        self.assertTrue(np.allclose(buffer.all([0, 1, 2])[0], buffer_reconstruct.all([0, 1, 2])[0]))
        self.assertTrue(np.allclose(buffer.all([0, 1, 2])[1], buffer_reconstruct.all([0, 1, 2])[1]))
        self.assertTrue(np.allclose(buffer.all([0, 1, 2])[2], buffer_reconstruct.all([0, 1, 2])[2]))
        self.assertTrue(np.allclose(buffer.all([0, 1, 2])[3], buffer_reconstruct.all([0, 1, 2])[3]))
        self.assertTrue(np.allclose(buffer.all([0, 1, 2])[4], buffer_reconstruct.all([0, 1, 2])[4]))

        shutil.rmtree(dir_name)

        self.assertEqual(buffer.max_len(), 10000)

    def test_transition_buffer(self):
        import rlutils
        import numpy as np
        env = rlutils.environment.PuddleWorld()
        policy = rlutils.policy.uniform_random(4)
        buffer_1 = rlutils.data.TransitionBuffer()
        rlutils.data.simulate(env, policy, buffer_1)
        self.assertEqual(buffer_1.max_len(), len(buffer_1.all()[1]))

        buffer_2 = rlutils.data.TransitionBuffer()
        rlutils.data.simulate(env, policy, buffer_2)

        buffer = buffer_1 + buffer_2
        self.assertTrue(np.allclose(buffer.all()[0], np.concatenate((buffer_1.all()[0], buffer_2.all()[0]), axis=0)))
        self.assertTrue(np.allclose(buffer.all()[1], np.concatenate((buffer_1.all()[1], buffer_2.all()[1]), axis=0)))
        self.assertTrue(np.allclose(buffer.all()[2], np.concatenate((buffer_1.all()[2], buffer_2.all()[2]), axis=0)))
        self.assertTrue(np.allclose(buffer.all()[3], np.concatenate((buffer_1.all()[3], buffer_2.all()[3]), axis=0)))
        self.assertTrue(np.allclose(buffer.all()[4], np.concatenate((buffer_1.all()[4], buffer_2.all()[4]), axis=0)))
