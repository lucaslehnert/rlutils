#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import os
import os.path as osp

import numpy as np
import yaml

from .simulate import TransitionListener


class TransitionBuffer(TransitionListener):
    """
    TransitionBuffer maintains a transition data set.
    """

    def __init__(self,
                 s_buffer=None,
                 a_buffer=None,
                 r_buffer=None,
                 s_next_buffer=None,
                 t_buffer=None,
                 info_buffer=None):
        self._s_buffer = s_buffer if s_buffer is not None else []
        self._a_buffer = a_buffer if a_buffer is not None else []
        self._r_buffer = r_buffer if r_buffer is not None else []
        self._s_next_buffer = s_next_buffer if s_next_buffer is not None else []
        self._t_buffer = t_buffer if t_buffer is not None else []
        self._info_buffer = info_buffer if info_buffer is not None else []

    def __len__(self):
        return len(self._a_buffer)

    def max_len(self):
        '''
        Maximum allowed length. If this is a buffer that is allowed to grow up to N transitions, then N is returned. If
        this buffer can grow without bound, then the current size is returned. In this case maximum_len behaves as
        __len__.

        :return:
        '''
        return len(self)

    def __add__(self, other):
        s_1, a_1, r_1, s_n_1, t_1, i_1 = self.all()
        s_2, a_2, r_2, s_n_2, t_2, i_2 = other.all()

        b_s = np.concatenate((s_1, s_2), axis=0)
        b_a = np.concatenate((a_1, a_2), axis=0)
        b_r = np.concatenate((r_1, r_2), axis=0)
        b_s_n = np.concatenate((s_n_1, s_n_2), axis=0)
        b_t = np.concatenate((t_1, t_2), axis=0)
        b_i = i_1 + i_2

        return TransitionBuffer(b_s, b_a, b_r, b_s_n, b_t, b_i)

    def remove_transition(self, idx=0):
        self._s_buffer.pop(idx)
        self._a_buffer.pop(idx)
        self._r_buffer.pop(idx)
        self._s_next_buffer.pop(idx)
        self._t_buffer.pop(idx)
        self._info_buffer.pop(idx)

    def all(self, idx_list=None):
        if idx_list is None:
            return np.array(self._s_buffer), \
                   np.array(self._a_buffer), \
                   np.array(self._r_buffer), \
                   np.array(self._s_next_buffer), \
                   np.array(self._t_buffer), \
                   self._info_buffer
        else:
            return np.array([self._s_buffer[i] for i in idx_list]), \
                   np.array([self._a_buffer[i] for i in idx_list]), \
                   np.array([self._r_buffer[i] for i in idx_list]), \
                   np.array([self._s_next_buffer[i] for i in idx_list]), \
                   np.array([self._t_buffer[i] for i in idx_list]), \
                   [self._info_buffer[i] for i in idx_list]

    def update_transition(self, s, a, r, s_next, t, info):
        self._s_buffer.append(s)
        self._a_buffer.append(a)
        self._r_buffer.append(r)
        self._s_next_buffer.append(s_next)
        self._t_buffer.append(t)
        self._info_buffer.append(info)

    def save(self, dir_name):
        os.makedirs(dir_name, exist_ok=True)
        np.save(osp.join(dir_name, 'buffer_s.npy'), self._s_buffer)
        np.save(osp.join(dir_name, 'buffer_a.npy'), self._a_buffer)
        np.save(osp.join(dir_name, 'buffer_r.npy'), self._r_buffer)
        np.save(osp.join(dir_name, 'buffer_s_next.npy'), self._s_next_buffer)
        np.save(osp.join(dir_name, 'buffer_t.npy'), self._t_buffer)
        with open(osp.join(dir_name, 'buffer_info.yaml'), 'w') as f:
            yaml.dump(self._info_buffer, f)

    def load(self, dir_name):
        self._s_buffer = np.load(osp.join(dir_name, 'buffer_s.npy'))
        self._a_buffer = np.load(osp.join(dir_name, 'buffer_a.npy'))
        self._r_buffer = np.load(osp.join(dir_name, 'buffer_r.npy'))
        self._s_next_buffer = np.load(osp.join(dir_name, 'buffer_s_next.npy'))
        self._t_buffer = np.load(osp.join(dir_name, 'buffer_t.npy'))
        with open(osp.join(dir_name, 'buffer_info.yaml'), 'r') as f:
            self._info_buffer = yaml.load(f, Loader=yaml.FullLoader)


class TransitionBufferFixedSize(TransitionBuffer):
    """
    TransitionBuffer maintains a transition data set of a fixed size.
    """

    def __init__(self,
                 num_transitions,
                 s_buffer=None,
                 a_buffer=None,
                 r_buffer=None,
                 s_next_buffer=None,
                 t_buffer=None,
                 info_buffer=None):
        super().__init__(s_buffer, a_buffer, r_buffer, s_next_buffer, t_buffer, info_buffer)
        self._num_transitions = num_transitions

    def update_transition(self, s, a, r, s_next, t, info):
        super().update_transition(s, a, r, s_next, t, info)
        if len(self) > self._num_transitions:
            super().remove_transition(0)

    def full(self):
        return len(self) >= self._num_transitions

    def max_len(self):
        return self._num_transitions
