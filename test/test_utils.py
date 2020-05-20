#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import unittest


class TestUtils(unittest.TestCase):

    def test_one_hot(self):
        import rlutils as rl
        import numpy as np
        self.assertTrue(np.all(rl.one_hot(0, 2) == np.array([1., 0.])))
        self.assertTrue(np.all(rl.one_hot(1, 2) == np.array([0., 1.])))
        self.assertTrue(np.all(rl.one_hot(0, 4) == np.array([1., 0., 0., 0.])))
        self.assertTrue(np.all(rl.one_hot(1, 4) == np.array([0., 1., 0., 0.])))
        self.assertTrue(np.all(rl.one_hot(2, 4) == np.array([0., 0., 1., 0.])))
        self.assertTrue(np.all(rl.one_hot(3, 4) == np.array([0., 0., 0., 1.])))

    def test_repeat_function_with_ndarray_return(self):
        import rlutils as rl
        import numpy as np
        ar_stack = rl.repeat_function_with_ndarray_return(lambda: np.ones(3, dtype=np.float32), 20, dtype=np.float32)
        self.assertEqual(ar_stack().dtype, np.float32)
        self.assertTrue(np.all(ar_stack() == np.ones([20, 3], dtype=np.float32)))
        ar_stack = rl.repeat_function_with_ndarray_return(lambda: np.ones(3, dtype=np.int32), 4, dtype=np.int32)
        self.assertEqual(ar_stack().dtype, np.int32)
        self.assertTrue(np.all(ar_stack() == np.ones([4, 3], dtype=np.int32)))

    def test_Experiment(self):
        import rlutils as rl
        import numpy as np
        import time
        import random

        class TestExperiment(rl.Experiment):
            def __init__(self):
                super().__init__()

            def _run_experiment(self):
                self.int = random.randint(1, 3)
                self.flt = np.random.randint(0, 2)
                time.sleep(2)

            def save(self):
                return self._duration_sec

        rl.set_seeds(12345)
        exp = TestExperiment()
        exp.run()
        duration = exp.save()
        self.assertEqual(2, exp.int)
        self.assertEqual(0, exp.flt)
        self.assertLessEqual(2., duration)
