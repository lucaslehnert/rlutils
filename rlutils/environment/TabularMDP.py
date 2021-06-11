#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import os
import os.path as osp

import gym
import numpy as np
import yaml
from gym.spaces import Discrete

from typing import List, Tuple
from rlutils.utils import one_hot
from .gridworld import add_terminal_states


class TabularMDP(gym.Env):
    ONE_HOT = 'one_hot'

    def __init__(
            self, 
            t_mat: np.ndarray, 
            r_mat: np.ndarray, 
            idx_start_list: List[int], 
            idx_goal_list: List[int], 
            name: str='TabularMDP'):
        num_states = np.shape(t_mat)[1]
        term_state_mask = np.array(
            [i in idx_goal_list for i in range(num_states)], dtype=bool
        )
        t_mat, r_mat = add_terminal_states(t_mat, r_mat, term_state_mask)

        self._t_mat = t_mat
        self._r_mat = r_mat

        self._idx_start_list = idx_start_list
        self._idx_goal_list = idx_goal_list

        self._s = self._idx_to_state(np.random.choice(self._idx_start_list))
        num_actions, _, _ = np.shape(self._t_mat)
        self.action_space = Discrete(num_actions)

        self._name = name

    def start_state_list(self) -> np.ndarray:
        return np.array(self._idx_start_list, copy=True)

    def goal_state_list(self) -> np.ndarray:
        return np.array(self._idx_goal_list, copy=True)

    def _idx_to_state(self, i):
        _, n, _ = np.shape(self._t_mat)
        return one_hot(i, n)

    def _state_to_idx(self, s):
        return np.where(s == 1.)[0][0]

    def _augment_state_dict(self, s: dict) -> dict:
        """Overload this method to implement manipulations to the state 
        dictionary. By default this method just returns its input.

        Args:
            s (dict): State dictionary.

        Returns:
            dict: Manipulated state dictionary.
        """
        return s

    def _wrap_in_state_dict(self, s: np.ndarray) -> dict:
        return self._augment_state_dict({TabularMDP.ONE_HOT: s})

    def reset(self, idx_start=None) -> dict:
        if idx_start is None:
            idx_start = np.random.choice(self._idx_start_list)
        self._s = self._idx_to_state(idx_start)
        return self._wrap_in_state_dict(np.copy(self._s))

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        s_prob = np.matmul(self._s, self._t_mat[action])
        s_ind_next = np.random.choice(np.arange(len(s_prob)), p=s_prob)
        r = np.matmul(self._s, self._r_mat[action])[s_ind_next]
        self._s *= 0
        self._s[s_ind_next] = 1.
        if self._state_to_idx(self._s) in self._idx_goal_list:
            done = True
        else:
            done = False
        return self._wrap_in_state_dict(np.copy(self._s)), r, done, {}

    def get_t_mat_r_mat(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.copy(self._t_mat), np.copy(self._r_mat)

    def get_t_mat_r_vec(self) -> Tuple[np.ndarray, np.ndarray]:
        r_vec = np.sum(self._t_mat * self._r_mat, axis=-1)
        return np.copy(self._t_mat), r_vec

    def render(self, mode='human', close='False'):  # pragma: no cover
        pass

    def num_states(self) -> int:
        return np.shape(self._t_mat)[1]

    def num_actions(self) -> int:
        return np.shape(self._t_mat)[0]

    def __str__(self) -> str:
        return self._name

    def save_to_file(self, meta_filename: str):
        """Save MDP to YAML file.

        Args:
            meta_filename (str): File name (include .yaml extension in string)
        """
        save_dir = osp.split(meta_filename)[0]
        os.makedirs(save_dir, exist_ok=True)
        fn_base = osp.splitext(osp.split(meta_filename)[1])[0]

        t_mat_fn = '{}_t_mat.npy'.format(fn_base)
        np.save(osp.join(save_dir, t_mat_fn), self._t_mat)
        r_mat_fn = '{}_r_mat.npy'.format(fn_base)
        np.save(osp.join(save_dir, r_mat_fn), self._r_mat)
        idx_start_list_fn = '{}_idx_start_list.npy'.format(fn_base)
        np.save(osp.join(save_dir, idx_start_list_fn), self._idx_start_list)
        idx_goal_list_fn = '{}_idx_goal_list.npy'.format(fn_base)
        np.save(
            osp.join(save_dir, f'{fn_base}_idx_goal_list.npy'), 
            self._idx_goal_list
        )

        mdp_dict = {
            'name': str(self),
            't_mat': t_mat_fn,
            'r_mat': r_mat_fn,
            'idx_start_list': idx_start_list_fn,
            'idx_goal_list': idx_goal_list_fn
        }
        with open(meta_filename, 'w') as f:
            yaml.dump(mdp_dict, f, default_flow_style=False)

    @classmethod
    def load_from_file(self, meta_filename: str):
        """Load MDP from file.

        Args:
            meta_filename (str): File name (include .yaml extension in string)

        Returns:
            [TabularMDP]: MDP object.
        """
        with open(meta_filename, 'r') as f:
            mdp_dict = yaml.load(f, Loader=yaml.Loader)
        save_dir = osp.split(meta_filename)[0]

        t_mat = np.load(osp.join(save_dir, mdp_dict['t_mat']))
        r_mat = np.load(osp.join(save_dir, mdp_dict['r_mat']))
        idx_start_list = np.load(osp.join(save_dir, mdp_dict['idx_start_list']))
        idx_goal_list = np.load(osp.join(save_dir, mdp_dict['idx_goal_list']))
        mdp = TabularMDP(
            t_mat, r_mat, idx_start_list, idx_goal_list, name=mdp_dict['name']
        )
        return mdp



class TabularMDPException(Exception):
    pass