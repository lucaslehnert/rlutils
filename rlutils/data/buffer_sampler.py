#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
from .replaybuffer import ReplayBuffer


class BufferSamplerUniform(object):
    """
    BufferSamplerUniform implements an iterator to sample uniformly from a 
    transition buffer. The iterator will stop after the specified number of 
    samples.
    """
    def __init__(self, buffer: ReplayBuffer, batch_size: int, num_samples: int):
        '''

        :param buffer: Transition buffer to sample from.
        :param batch_size: Batch size.
        :param num_samples: Number of samples drawn from buffer. If none, then 
            the buffer will draw samples forever (infinite loop).
        '''
        self._buffer = buffer
        self._batch_size = batch_size
        self._num_sampl = num_samples
        self._cnt_sampl = 0
        self._idx_list = np.arange(self._buffer.max_len())

    def __iter__(self):
        return self

    def __next__(self):
        if self._num_sampl is not None and self._cnt_sampl >= self._num_sampl:
            raise StopIteration
        if len(self._buffer) < self._batch_size:
            batch = self._batch_size
            n = self._buffer.num_transitions()
            raise BufferIteratorException(
                f'Cannot sample {batch} transitions from buffer of size {n}.'
            )
        self._cnt_sampl += 1
        idx_list = np.random.choice(
            self._idx_list[:n], size=self._batch_size
        )
        return self._buffer.ge


class BufferIteratorException(Exception):
    pass


class BufferSamplerUniformSARST(BufferSamplerUniform):
    def __next__(self):
        return super().__next__()[:-1]


class BufferSamplerUniformSARS(BufferSamplerUniform):
    def __next__(self):
        return super().__next__()[:-2]


class BufferIterator(object):
    """BufferIterator implements an iterator to iterate in batches of 
    transitions over a transition buffer. After all transitions are sampled 
    from the buffer, sampling will begin again at the beginning of the buffer. 

    """
    def __init__(self, buffer, batch_size, num_samples, shuffle=True):
        self._buffer = buffer
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._cnt_samples = 0
        self._idx_list = np.arange(len(self._buffer))
        self._shuffle = shuffle
        if self._shuffle:
            np.random.shuffle(self._idx_list)
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._cnt_samples >= self._num_samples:
            raise StopIteration
        self._cnt_samples += 1

        idx_list = []
        for i in range(self._batch_size):
            idx_list.append((self._i + i) % len(self._buffer))
        self._i = idx_list[-1] + 1

        if self._shuffle:
            idx_list = np.random.choice(self._idx_list, size=self._batch_size)
        return self._buffer.all(idx_list=idx_list)


class BufferIteratorSARST(BufferIterator):
    def __next__(self):
        return super().__next__()[:-1]


class BufferIteratorSARS(BufferIterator):
    def __next__(self):
        return super().__next__()[:-2]
