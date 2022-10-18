#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import random
import time
from abc import abstractmethod
from datetime import timedelta
from typing import Callable, Any

import numpy as np


def set_seeds(seed: int):
    """
    Shorthand for setting random number seeds for the libraries numpy and random.

    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)


def one_hot(i: int, n: int, dtype=np.float32):
    """
    Construct a one-hot bit vector stored in numpy.

    :param i: Index at which a one is inserted into one-hot bit vector.
    :param n: Length of the constructed one-hot bit vector.
    :param dtype: dtype of constructed numpy array.
    :return: numpy array of length n containing one-hot bit vector.
    """
    v = np.zeros(n, dtype=dtype)
    v[i] = 1.
    return v


class Experiment:
    """
    Abstract experiment super class. An experiment can implement this super class.

    The actual experiment procedure should be implemented in the sub-class by implementing _run_experiment.

    The method save persists the experiment and its collected data. The method load reconstructs the experiment will
    all collected data.
    """

    def __init__(self):
        self._duration_sec = None

    def run(self):
        t_start = time.time()
        self._run_experiment()
        self._duration_sec = time.time() - t_start
        print('{} finished, duration: {:0>8}'.format(self.get_class_name(), str(timedelta(seconds=self._duration_sec))))

    @abstractmethod
    def _run_experiment(self):  # pragma: no cover
        pass

    @abstractmethod
    def save(self):  # pragma: no cover
        """
        Save experiment to file. The location has to be passed into the constructor.
        :return: None
        """
        pass

    @staticmethod
    def load(save_dir: str):  # pragma: no cover
        """
        Classmethod used to load an experiment from a meta file.
        :param save_dir: Directory into which experiment is persisted.
        :return: An instance of Experiment.
        """
        raise NotImplemented(
            'load_experiment must be implemented by a subclass.')

    @classmethod
    def get_class_name(cls):  # pragma: no cover
        return cls.__name__


class ExperimentException(Exception):  # pragma: no cover
    """
    Exception thrown by the Experiment class. This class is deprecated and will 
    be removed.
    """
    pass


def repeat_function_with_ndarray_return(
    func: Callable[..., np.ndarray], 
    repeats: int=20
) -> Callable[..., np.ndarray]:
    """Function decorator that repeats a function that returns a numpy array. 
    The returned numpy arrays are stacked and the stacked array is returned.

    Args:
        func (Callable[..., np.ndarray]): _description_
        repeats (int, optional): _description_. Defaults to 20.

    Returns:
        np.ndarray: _description_
    """

    def func_rep(*params, **kwargs):
        return np.array([func(*params, **kwargs) for _ in range(repeats)])

    return func_rep
