#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from __future__ import annotations
from abc import abstractmethod
import random
import numpy as np
import h5py
import os
import random
from typing import List, Dict, Optional


Action = b"action"
Reward = b"reward"
Term = b"term"


class ReplayBuffer:
    """This class describes the replay buffer interface.
    """

    @abstractmethod
    def get_transition_column_names(self) -> List[str]:
        """Transition column name getter.

        :return: Transition column names.
        :rtype: List[str]
        """
        pass

    @abstractmethod
    def get_state_column_names(self) -> List[str]:
        """State column name getter.

        :return: State column names.
        :rtype: List[str]
        """
        pass

    @abstractmethod
    def num_transitions(self) -> int:
        """Returns the number of available transitions. Transitions are indexed
        from 0 up to n - 1, where n is returned by ``num_transitions``.

        :return: Number of transitions.
        :rtype: int
        """
        pass

    @abstractmethod
    def num_states(self) -> int:
        """Returns the number of available states. States are indexed from 0 up 
        to n - 1, where n is returned by ``num_states``.

        :return: Number of states.
        :rtype: int
        """
        pass

    @abstractmethod
    def get_column(
        self,
        column_name: str,
        idxs: Optional[List[int]] = None
    ) -> np.ndarray:
        """Get values for the column with name ``column_name``. If transition 
        indices are provided, then only the value of the indexed transitions are
        returned. Here, ``column_name'' must be contained in the list returned 
        by ``get_transition_column_names`` otherwise an error is 
        thrown.

        :param column_name: Transition column name.
        :type column_name: str
        :param idxs: Transition indices, defaults to None
        :type idxs: Optional[List[int]], optional
        :return: Column values.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def set_column(
        self,
        column_name: str,
        column_value: np.ndarray,
        idxs: Optional[List[int]] = None
    ):
        """Set values for the column with name ``column_name``. If transition 
        indices are provided, then only the indexed transitions are updated. 
        Here, ``column_name`` must be contained in the list returned by 
        ``get_transition_column_names`` otherwise an error is thrown.

        :param column_name: Transition column name.
        :type column_name: str
        :param column_value: Value updates.
        :type column_value: np.ndarray
        :param idxs: Transition indices, defaults to None
        :type idxs: Optional[List[int]], optional
        """
        pass

    @abstractmethod
    def get_state_column(
        self,
        column_name: str,
        idxs: Optional[List[int]] = None
    ) -> np.ndarray:
        """Get state column values with name ``column_name`` for all states 
        contained in the replay buffer. If indices are provided, then only 
        values for the corresonding states are returned. Indices can range from
        0 to less than ``self.num_states()``. If indices are out of range, an
        assertion error is thrown.

        :param column_name: Column name
        :type column_name: str
        :param idxs: States indices, defaults to None.
        :type idxs: Optional[List[int]], optional
        :return: Column values.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def set_state_column(
        self,
        column_name: str,
        column_value: np.ndarray,
        idxs: Optional[List[int]] = None
    ):
        """Set values for the column with name ``column_name``. If state indices
        are provided, then only the indexed transitions are updated. Here, 
        ``column_name'' must be contained in the list returned by 
        ``get_state_column_names`` otherwise an error is thrown. If 
        indices are provided, then only values for the corresonding states are 
        returned. Indices can range from 0 to less than ``self.num_states()``. 
        If indices are out of range, an assertion error is thrown.

        :param column_name: Column name
        :type column_name: str
        :param column_value: State indices, defaults to None
        :type column_value: np.ndarray, optional
        """
        pass

    @abstractmethod
    def get_start_state_column(
        self,
        column_name: str,
        idxs: Optional[List[int]] = None
    ) -> np.ndarray:
        """Get state column values with name ``column_name`` for all states 
        that are at the start of some transition. If indices are provided, then 
        only start-state values for the corresonding transitions are returned. 
        Indices can range from 0 to less than ``self.num_transitions()``. If 
        indices are out of range, an assertion error is thrown.

        :param column_name: Column name
        :type column_name: str
        :param idxs: Transition indices, defaults to None.
        :type idxs: List[int], optional
        :return: Column values.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def set_start_state_column(
        self,
        column_name: str,
        column_value: np.ndarray,
        idxs: Optional[List[int]] = None
    ):
        """Set start-state values for the state column with name 
        ``column_name``. If state indices are provided, then only the indexed 
        transitions are updated. Here, ``column_name'' must be contained in the 
        list returned by ``get_state_column_names`` otherwise an error 
        is thrown. If indices are provided, then only values for the 
        corresonding states are returned. Indices can range from 0 to less than 
        ``self.num_states()``. If indices are out of range, an assertion error 
        is thrown.

        :param column_name: Column name
        :type column_name: str
        :param column_value: State indices, defaults to None
        :type column_value: np.ndarray, optional
        """
        pass

    @abstractmethod
    def get_next_state_column(
        self,
        column_name: str,
        idxs: Optional[List[int]] = None
    ) -> np.ndarray:
        """Get state column values with name ``column_name`` for all states 
        that are at the end of some transition. If indices are provided, then 
        only end-state values for the corresonding transitions are returned. 
        Indices can range from 0 to less than ``self.num_transitions()``. If 
        indices are out of range, an assertion error is thrown.

        :param column_name: Column name
        :type column_name: str
        :param idxs: Transsition indices, defaults to None.
        :type idxs: List[int], optional
        :return: Column values.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def set_next_state_column(
        self,
        column_name: str,
        column_value: np.ndarray,
        idxs:  Optional[List[int]] = None
    ):
        """Set end-state values for the state column with name ``column_name``. 
        If state indices are provided, then only the indexed transitions are 
        updated. Here, ``column_name'' must be contained in the list returned by
        ``get_state_column_names`` otherwise an error is thrown. If 
        indices are provided, then only values for the corresonding states are 
        returned. Indices can range from 0 to less than ``self.num_states()``. 
        If indices are out of range, an assertion error is thrown.

        :param column_name: Column name
        :type column_name: str
        :param column_value: State indices, defaults to None
        :type column_value: np.ndarray, optional
        """
        pass

    @abstractmethod
    def add_transition(
        self,
        state: Dict[str, np.ndarray],
        transition: Dict[str, np.ndarray],
        next_state: Dict[str, np.ndarray]
    ):
        """Add a transition to the replay buffer. A transition consists of three
        different dictionaries: A ``state`` dictionary describing the start
        state, a ``transition`` dictionary describing the transition (e.g.
        actions, rewards, and other fields), and the ``next_state`` dictionary 
        describing the next state.

        :param state: Start state
        :type state: Dict[str, np.ndarray]
        :param transition_prop: Transition properties
        :type transition_prop: Dict[str, np.ndarray]
        :param next_state: Next state
        :type next_state: Dict[str, np.ndarray]
        """
        pass

    @abstractmethod
    def finish_current_sequence(self):
        """Signals the end of a trajectory. The next-added transition will be 
        considered as a trajectory start. Calling this method is equivalent to
        calling ``add_transition`` with ``finish_sequence=True`` when adding the
        previous transition.

        Calling this method multiple times in a row without adding a transition 
        to the replay buffer causes a ``ReplayBufferException'' to be thrown.
        """
        pass

    def filter_transition_column(self, column_names: List[str]) -> ReplayBuffer:
        """Slices the replay buffer by transition columns. The returned
        ``ReplayBuffer`` object contains only the specified transition columns.
        This object references into the original replay buffer and does not
        duplicate any data. Instead, the replay buffer only provides a
        re-indexed view. Consequently, any modifications to the returned
        ``ReplayBuffer`` also effect the state of the orignal replay buffer.

        :param column_names: Column names that remain visible.
        :type column_names: List[str]
        :return: [description]
        :rtype: ReplayBuffer
        """
        return ReplayBufferTransitionColumnWrapper(self, column_names)

    def filter_state_column(self, column_names: List[str]) -> ReplayBuffer:
        """Slices the replay buffer by state columns. The returned
        ``ReplayBuffer`` object contains only the specified state columns.This
        object references into the original replay buffer and does not
        duplicate any data. Instead, the replay buffer only provides a
        re-indexed view. Consequently, any modifications to the returned
        ``ReplayBuffer`` also effect the state of the orignal replay buffer.

        :param column_names: Column names that remain visible.
        :type column_names: List[str]
        :return: [description]
        :rtype: ReplayBuffer
        """
        return ReplayBufferStateColumnWrapper(self, column_names)


class ReplayBufferException(Exception):
    pass


class Trajectory(ReplayBuffer):
    def __init__(
        self,
        transition_spec: 
        state_defaults: Dict[str, np.ndarray],
        transition_defaults: Dict[str, np.ndarray]
    ):
        if len(state_defaults) == 0:
            raise ReplayBufferException(
                'Cannot construct trajectory with empty state columns.')
        if len(transition_defaults) == 0:
            raise ReplayBufferException(
                'Cannot construct trajectory with empty transition columns.')

        self._state_defaults = state_defaults
        self._transition_defaults = transition_defaults
        self._state_col: Dict[str, ] = {k: [] for k in state_defaults.keys()}
        self._transition_col = {k: [] for k in transition_defaults.keys()}

    def persist_columns_in_h5_group(
        self,
        h5grp,
        state_columns: Optional[List[str]] = None,
        transition_columns: Optional[List[str]] = None
    ):
        if state_columns is None:
            state_columns = self._state_col.keys()
        if transition_columns is None:
            transition_columns = self._transition_col.keys()

        state_grp = h5grp.create_group('state_col')
        for k in state_columns:
            v = self._state_col.get(k)
            col_grp = state_grp.create_group(k)
            col_grp.create_dataset('value', data=np.reshape(v, -1))
            col_grp.create_dataset('shape', data=np.shape(v))
        transition_grp = h5grp.create_group('transition_col')
        for k in transition_columns:
            v = self._transition_col.get(k)
            col_grp = transition_grp.create_group(k)
            col_grp.create_dataset('value', data=np.reshape(v, -1))
            col_grp.create_dataset('shape', data=np.shape(v))

    @staticmethod
    def from_h5_group(h5grp, state_defaults, transition_defaults):
        traj = Trajectory(state_defaults, transition_defaults)
        for k, v in h5grp['state_col'].items():
            v_np = np.array(v['value'])
            v_np = np.reshape(v_np, v['shape'])
            v_np = [e for e in v_np]
            traj._state_col[k] += v_np
        for k, v in h5grp['transition_col'].items():
            v_np = np.array(v['value'])
            v_np = np.reshape(v_np, v['shape'])
            v_np = [e for e in v_np]
            traj._transition_col[k] += v_np
        return traj

    @staticmethod
    def from_h5_file(filename, groupname, state_defaults, transition_defaults):
        with h5py.File(filename, 'r') as hf:
            grp = hf['trajectories'][groupname]
            traj = Trajectory.from_h5_group(
                grp, state_defaults, transition_defaults
            )
        return traj

    def add_transition(
            self,
            state: Dict[str, np.ndarray],
            transition: Dict[str, np.ndarray],
            next_state: Dict[str, np.ndarray]):
        if self.num_transitions() == 0:
            for k, v_default in self._state_defaults.items():
                v = state.get(k, v_default)
                v = np.array(v, copy=True)
                self._state_col[k].append(v)
        for k, v_default in self._transition_defaults.items():
            v = transition.get(k, v_default)
            v = np.array(v, copy=True)
            self._transition_col[k].append(v)
        for k, v_default in self._state_defaults.items():
            v = next_state.get(k, v_default)
            v = np.array(v, copy=True)
            self._state_col[k].append(v)

    def finish_current_sequence(self):
        """Calling this method has no effect on a trajectory.
        """
        pass

    def get_transition_column_names(self) -> List[str]:
        return list(self._transition_col.keys())

    def get_state_column_names(self) -> List[str]:
        return list(self._state_col.keys())

    def num_transitions(self) -> int:
        k = self.get_transition_column_names()[0]
        return len(self._transition_col[k])

    def num_states(self) -> int:
        k = self.get_state_column_names()[0]
        return len(self._state_col[k])

    def get_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if idxs is None:
            col = self._transition_col[column_name]
        else:
            col = [self._transition_col[column_name][i] for i in idxs]
        return np.array(col, copy=True)

    def set_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if idxs is None:
            self._transition_col[column_name][:] = column_value
        else:
            for i, c in zip(idxs, column_value):
                self._transition_col[column_name][i] = c

    def get_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if idxs is None:
            col = self._state_col[column_name]
        else:
            col = [self._state_col[column_name][i] for i in idxs]
        return np.array(col, copy=True)

    def set_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if idxs is None:
            self._state_col[column_name][:] = column_value
        else:
            for i, v in zip(idxs, column_value):
                self._state_col[column_name][i] = v

    def get_start_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if idxs is None:
            col = self._state_col[column_name][:-1]
        else:
            col = [self._state_col[column_name][i] for i in idxs]
        return np.array(col, copy=True)

    def set_start_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if idxs is None:
            self._state_col[column_name][:-1] = column_value
        else:
            for i, v in zip(idxs, column_value):
                self._state_col[column_name][i] = v

    def get_next_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        col = self._state_col[column_name][1:]
        if idxs is not None:
            col = [col[i] for i in idxs]
        return np.array(col, copy=True)

    def set_next_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if idxs is None:
            self._state_col[column_name][1:] = column_value
        else:
            for i, v in zip(idxs, column_value):
                self._state_col[column_name][i + 1] = v


class TrajectoryBuffer(ReplayBuffer):
    def __init__(
            self,
            state_defaults: Dict[str, np.ndarray],
            transition_defaults: Dict[str, np.ndarray]):
        self._state_defaults = state_defaults
        self._transition_defaults = transition_defaults

        self._trajectories = [Trajectory(state_defaults, transition_defaults)]
        self._traj_idxs = []
        self._step_idxs = []
        self._state_traj_idxs = []
        self._state_step_idxs = []

    def save_h5(
            self,
            filename: str,
            state_columns: List[str] = None,
            transition_columns: List[str] = None):
        dirname = os.path.split(os.path.abspath(filename))[0]
        os.makedirs(dirname, exist_ok=True)

        with h5py.File(filename, 'w') as hf:
            traj_grp = hf.create_group('trajectories')
            for i, t in enumerate(self._trajectories):
                traj_grp_i = traj_grp.create_group(f'{i}')
                t.persist_columns_in_h5_group(
                    traj_grp_i, state_columns, transition_columns
                )

            state_defaults_grp = hf.create_group('state_defaults')
            for k, v in self._state_defaults.items():
                grp = state_defaults_grp.create_group(k)
                grp.create_dataset('value', data=np.reshape(v, -1))
                grp.create_dataset('shape', data=np.shape(v))

            transition_defaults_grp = hf.create_group('transition_defaults')
            for k, v in self._transition_defaults.items():
                grp = transition_defaults_grp.create_group(k)
                grp.create_dataset('value', data=np.reshape(v, -1))
                grp.create_dataset('shape', data=np.shape(v))

            hf.create_dataset(
                'traj_idxs', data=np.array(self._traj_idxs)
            )
            hf.create_dataset(
                'step_idxs', data=np.array(self._step_idxs)
            )
            hf.create_dataset(
                'state_traj_idxs', data=np.array(self._state_traj_idxs)
            )
            hf.create_dataset(
                'state_step_idxs', data=np.array(self._state_step_idxs)
            )

    @staticmethod
    def load_h5(filename: str) -> TrajectoryBuffer:
        with h5py.File(filename, 'r') as hf:
            s_dflts = {}
            for k, v in hf['state_defaults'].items():
                v_np = np.array(v['value'])
                v_np = np.reshape(v_np, v['shape'])
                s_dflts[k] = v_np
            t_dflts = {}
            for k, v in hf['transition_defaults'].items():
                v_np = np.array(v['value'])
                v_np = np.reshape(v_np, v['shape'])
                t_dflts[k] = v_np

            buffer = TrajectoryBuffer(s_dflts, t_dflts)
            buffer._traj_idxs = list(np.array(hf['traj_idxs']))
            buffer._step_idxs = list(np.array(hf['step_idxs']))
            buffer._state_traj_idxs = list(np.array(hf['state_traj_idxs']))
            buffer._state_step_idxs = list(np.array(hf['state_step_idxs']))

            traj_keys = [f'{i}' for i in range(len(hf['trajectories'].keys()))]

        params = [(filename, k, s_dflts, t_dflts) for k in traj_keys]
        buffer._trajectories = [Trajectory.from_h5_file(*p) for p in params]
        # with mp.Pool() as p:
        #     buffer._trajectories = p.starmap(Trajectory.from_h5_file, params)

        # buffer._trajectories = []
        # for k in traj_keys:
        #     buffer._trajectories.append(Trajectory.from_h5_file(
        #         filename, k, s_dflts, t_dflts
        #     ))
        # traj_dict = {}
        # for i, t in hf['trajectories'].items():
        #     traj = Trajectory.from_h5_group(
        #         t, state_defaults, transition_defaults
        #     )
        #     traj_dict[i] = traj
        # num_traj = len(traj_dict)
        # buffer._trajectories = [traj_dict[f'{i}'] for i in range(num_traj)]
        return buffer

    def get_transition_column_names(self) -> List[str]:
        return self._trajectories[0].get_transition_column_names()

    def get_state_column_names(self) -> List[str]:
        return self._trajectories[0].get_state_column_names()

    def num_transitions(self) -> int:
        return len(self._traj_idxs)

    def num_states(self) -> int:
        return len(self._state_traj_idxs)

    def get_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if idxs is None:
            traj_idxs = self._traj_idxs
            step_idxs = self._step_idxs
        else:
            traj_idxs = [self._traj_idxs[i] for i in idxs]
            step_idxs = [self._step_idxs[i] for i in idxs]
        res = []
        for t, s in zip(traj_idxs, step_idxs):
            r = self._trajectories[t].get_column(column_name, idxs=[s])[0]
            res.append(r)
        return np.array(res)

    def set_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if idxs is None:
            traj_idxs = self._traj_idxs
            step_idxs = self._step_idxs
        else:
            traj_idxs = [self._traj_idxs[i] for i in idxs]
            step_idxs = [self._step_idxs[i] for i in idxs]
        for t, s, v in zip(traj_idxs, step_idxs, column_value):
            self._trajectories[t].set_column(column_name, [v], idxs=[s])

    def get_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if idxs is None:
            traj_idxs = self._state_traj_idxs
            step_idxs = self._state_step_idxs
        else:
            traj_idxs = [self._state_traj_idxs[i] for i in idxs]
            step_idxs = [self._state_step_idxs[i] for i in idxs]
        res = []
        for t, s in zip(traj_idxs, step_idxs):
            r = self._trajectories[t].get_state_column(
                column_name, idxs=[s])[0]
            res.append(r)
        return np.array(res)

    def set_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if idxs is None:
            traj_idxs = self._state_traj_idxs
            step_idxs = self._state_step_idxs
        else:
            traj_idxs = [self._state_traj_idxs[i] for i in idxs]
            step_idxs = [self._state_step_idxs[i] for i in idxs]
        for t, s, v in zip(traj_idxs, step_idxs, column_value):
            self._trajectories[t].set_state_column(column_name, [v], idxs=[s])

    def get_start_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if idxs is None:
            traj_idxs = self._traj_idxs
            step_idxs = self._step_idxs
        else:
            traj_idxs = [self._traj_idxs[i] for i in idxs]
            step_idxs = [self._step_idxs[i] for i in idxs]
        res = []
        for t, s in zip(traj_idxs, step_idxs):
            traj = self._trajectories[t]
            r = traj.get_start_state_column(column_name, idxs=[s])[0]
            res.append(r)
        return np.array(res)

    def set_start_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if idxs is None:
            traj_idxs = self._traj_idxs
            step_idxs = self._step_idxs
        else:
            traj_idxs = [self._traj_idxs[i] for i in idxs]
            step_idxs = [self._step_idxs[i] for i in idxs]
        for t, s, v in zip(traj_idxs, step_idxs, column_value):
            traj = self._trajectories[t]
            traj.set_start_state_column(column_name, [v], idxs=[s])

    def get_next_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if idxs is None:
            traj_idxs = self._traj_idxs
            step_idxs = self._step_idxs
        else:
            traj_idxs = [self._traj_idxs[i] for i in idxs]
            step_idxs = [self._step_idxs[i] for i in idxs]
        res = []
        for t, s in zip(traj_idxs, step_idxs):
            traj = self._trajectories[t]
            r = traj.get_next_state_column(column_name, idxs=[s])[0]
            res.append(r)
        return np.array(res)

    def set_next_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if idxs is None:
            traj_idxs = self._traj_idxs
            step_idxs = self._step_idxs
        else:
            traj_idxs = [self._traj_idxs[i] for i in idxs]
            step_idxs = [self._step_idxs[i] for i in idxs]
        for t, s, v in zip(traj_idxs, step_idxs, column_value):
            traj = self._trajectories[t]
            traj.set_next_state_column(column_name, [v], idxs=[s])

    def add_transition(
            self,
            state: Dict[str, np.ndarray],
            transition: Dict[str, np.ndarray],
            next_state: Dict[str, np.ndarray]):
        if self._trajectories[-1].num_states() == 0:
            self._state_traj_idxs.append(len(self._trajectories) - 1)
            self._state_step_idxs.append(0)
        self._trajectories[-1].add_transition(state, transition, next_state)
        self._traj_idxs.append(len(self._trajectories) - 1)
        self._step_idxs.append(self._trajectories[-1].num_transitions() - 1)
        self._state_traj_idxs.append(len(self._trajectories) - 1)
        self._state_step_idxs.append(self._trajectories[-1].num_states() - 1)

    def finish_current_sequence(self):
        self._trajectories.append(
            Trajectory(self._state_defaults, self._transition_defaults)
        )


class TrajectoryBufferFixedSize(TrajectoryBuffer):
    def __init__(
            self,
            state_defaults: Dict[str, np.ndarray],
            transition_defaults: Dict[str, np.ndarray],
            max_transitions: int):
        super().__init__(state_defaults, transition_defaults)
        self.max_transitions = max_transitions

    def add_transition(
            self,
            state: Dict[str, np.ndarray],
            transition: Dict[str, np.ndarray],
            next_state: Dict[str, np.ndarray]):
        super().add_transition(state, transition, next_state)

        '''
        Adding a transition may cause the replay buffer to contain one 
        transition too many. The code below removes this transition.
        '''
        if len(self._traj_idxs) <= self.max_transitions:
            return

        self._traj_idxs = self._traj_idxs[1:]
        self._step_idxs = self._step_idxs[1:]
        self._state_traj_idxs = self._state_traj_idxs[1:]
        self._state_step_idxs = self._state_step_idxs[1:]
        if self._traj_idxs[0] == 1:
            self._trajectories = self._trajectories[1:]
            self._state_traj_idxs = self._state_traj_idxs[1:]
            self._state_step_idxs = self._state_step_idxs[1:]
            for i, v in enumerate(self._traj_idxs):
                self._traj_idxs[i] = v - 1
            for i, v in enumerate(self._state_traj_idxs):
                self._state_traj_idxs[i] = v - 1

    # def save_h5(self, filename: str):
    #     """Persist replay buffer into an H5 container.

    #     :param filename: Path to H5 file.
    #     :type filename: str
    #     """
    #     os.makedirs(os.path.split(os.path.abspath(filename))[0], exist_ok=True)
    #     with h5py.File(filename, 'w') as hf:
    #         state_grp = hf.create_group('state_col')
    #         for k, v in self._state_cols.items():
    #             col_grp = state_grp.create_group(k)
    #             col_grp.create_dataset('value', data=np.reshape(v, -1))
    #             col_grp.create_dataset('shape', data=np.shape(v))
    #         transition_grp = hf.create_group('transition_col')
    #         for k, v in self._transition_cols.items():
    #             col_grp = transition_grp.create_group(k)
    #             col_grp.create_dataset('value', data=np.reshape(v, -1))
    #             col_grp.create_dataset('shape', data=np.shape(v))
    #         state_defaults_grp = hf.create_group('state_defaults')
    #         for k, v in self._state_defaults.items():
    #             col_grp = state_defaults_grp.create_group(k)
    #             col_grp.create_dataset('value', data=np.reshape(v, -1))
    #             col_grp.create_dataset('shape', data=np.shape(v))
    #         transition_defaults_grp = hf.create_group('transition_defaults')
    #         for k, v in self._transition_defaults.items():
    #             col_grp = transition_defaults_grp.create_group(k)
    #             col_grp.create_dataset('value', data=np.reshape(v, -1))
    #             col_grp.create_dataset('shape', data=np.shape(v))
    #         hf.create_dataset('insert_idx', data=[self._insert_idx])
    #         hf.create_dataset('valid_idxs', data=self._valid_idxs)

    #         if self._max_num_transitions is not None:
    #             hf.create_dataset(
    #                 'max_num_transitions', data=[self._max_num_transitions])

    # @staticmethod
    # def load_h5(filename: str) -> NumpyReplayBuffer:
    #     """Load replay buffer from H5 container.

    #     :param filename: Path to H5 file.
    #     :type filename: str
    #     :return: Reconstructed replay buffer.
    #     :rtype: NumpyReplayBuffer
    #     """
    #     with h5py.File(filename, 'r') as hf:
    #         state_grp = hf['state_col']
    #         state_col = {}
    #         for k in state_grp.keys():
    #             ar = np.array(state_grp[k]['value'])
    #             ar = np.reshape(ar, np.array(state_grp[k]['shape']))
    #             state_col[k] = ar
    #         transition_grp = hf['transition_col']
    #         transition_col = {}
    #         for k in transition_grp.keys():
    #             ar = np.array(transition_grp[k]['value'])
    #             ar = np.reshape(ar, np.array(transition_grp[k]['shape']))
    #             transition_col[k] = ar
    #         state_def_grp = hf['state_defaults']
    #         state_def = {}
    #         for k in state_def_grp.keys():
    #             ar = np.array(state_def_grp[k]['value'])
    #             ar = np.reshape(ar, np.array(state_def_grp[k]['shape']))
    #             state_def[k] = ar
    #         transition_def_grp = hf['transition_defaults']
    #         transition_def = {}
    #         for k in transition_def_grp.keys():
    #             ar = np.array(transition_def_grp[k]['value'])
    #             ar = np.reshape(ar, np.array(transition_def_grp[k]['shape']))
    #             transition_def[k] = ar
    #         replay_buffer = NumpyReplayBuffer(
    #             state_columns=state_col,
    #             transition_columns=transition_col,
    #             state_defaults=state_def,
    #             transition_defaults=transition_def
    #         )
    #         replay_buffer._valid_idxs = list(hf['valid_idxs'])
    #         replay_buffer._insert_idx = list(hf['insert_idx'])[0]

    #         if 'max_num_transitions' in hf.keys():
    #             replay_buffer._max_num_transitions = list(
    #                 hf['max_num_transitions'])[0]
    #     return replay_buffer


class ReplayBufferTransitionColumnWrapper(ReplayBuffer):
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            transition_columns: List[str]):
        """Replay buffer transition column wrapper.

        :param replay_buffer: Replay buffer.
        :type replay_buffer: ReplayBuffer
        :param transition_columns: Transition column names to expose.
        :type transition_columns: List[str]
        """
        self._replay_buffer = replay_buffer
        self._transition_columns = transition_columns

    def get_transition_column_names(self) -> List[str]:
        return [n for n in self._transition_columns]

    def get_state_column_names(self) -> List[str]:
        return self._replay_buffer.get_state_column_names()

    def num_transitions(self) -> int:
        return self._replay_buffer.num_transitions()

    def num_states(self) -> int:
        return self._replay_buffer.num_states()

    def get_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if column_name not in self._transition_columns:
            raise ReplayBufferException(
                f'Column {column_name} is not visible.')
        return self._replay_buffer.get_column(column_name, idxs)

    def set_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if column_name not in self._transition_columns:
            raise ReplayBufferException(
                f'Column {column_name} it not visible.')
        self._replay_buffer.set_column(column_name, column_value, idxs)

    def get_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        return self._replay_buffer.get_state_column(column_name, idxs)

    def set_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        self._replay_buffer.set_state_column(column_name, column_value, idxs)

    def get_start_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        return self._replay_buffer.get_state_column(column_name, idxs)

    def set_start_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        self._replay_buffer.set_start_state_column(
            column_name, column_value, idxs)

    def get_next_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        return self._replay_buffer.get_next_state_column(column_name, idxs)

    def set_next_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        self._replay_buffer.set_next_state_column(
            column_name, column_name, idxs)

    def add_transition(
            self,
            state: Dict[str, np.ndarray],
            transition: Dict[str, np.ndarray],
            next_state: Dict[str, np.ndarray]):
        for k in transition.keys():
            if k not in self._transition_columns:
                raise ReplayBufferException(f'Column {k} it not visible.')
        self._replay_buffer.add_transition(state, transition, next_state)

    def finish_current_sequence(self):
        self._replay_buffer.finish_current_sequence()


class ReplayBufferStateColumnWrapper(ReplayBuffer):
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            state_columns: List[str]):
        """Replay buffer transition column wrapper.

        :param replay_buffer: Replay buffer.
        :type replay_buffer: ReplayBuffer
        :param state_columns: State column names to expose.
        :type state_columns: List[str]
        """
        self._replay_buffer = replay_buffer
        self._state_columns = state_columns

    def get_transition_column_names(self) -> List[str]:
        return self._replay_buffer.get_state_column_names()

    def get_state_column_names(self) -> List[str]:
        return [n for n in self._state_columns]

    def num_transitions(self) -> int:
        return self._replay_buffer.num_transitions()

    def num_states(self) -> int:
        return self._replay_buffer.num_states()

    def get_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        return self._replay_buffer.get_column(column_name, idxs)

    def set_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        self._replay_buffer.set_column(column_name, column_value, idxs)

    def get_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if column_name not in self._state_columns:
            raise ReplayBufferException(
                f'Column {column_name} it not visible.')
        return self._replay_buffer.get_state_column(column_name, idxs)

    def set_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if column_name not in self._state_columns:
            raise ReplayBufferException(
                f'Column {column_name} it not visible.')
        self._replay_buffer.set_state_column(column_name, column_value, idxs)

    def get_start_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if column_name not in self._state_columns:
            raise ReplayBufferException(
                f'Column {column_name} it not visible.')
        return self._replay_buffer.get_state_column(column_name, idxs)

    def set_start_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if column_name not in self._state_columns:
            raise ReplayBufferException(
                f'Column {column_name} it not visible.')
        self._replay_buffer.set_start_state_column(
            column_name, column_value, idxs)

    def get_next_state_column(
            self,
            column_name: str,
            idxs: List[int] = None) -> np.ndarray:
        if column_name not in self._state_columns:
            raise ReplayBufferException(
                f'Column {column_name} it not visible.')
        return self._replay_buffer.get_next_state_column(column_name, idxs)

    def set_next_state_column(
            self,
            column_name: str,
            column_value: np.ndarray,
            idxs: List[int] = None):
        if column_name not in self._state_columns:
            raise ReplayBufferException(
                f'Column {column_name} it not visible.')
        self._replay_buffer.set_next_state_column(
            column_name, column_name, idxs)

    def add_transition(
            self,
            state: Dict[str, np.ndarray],
            transition: Dict[str, np.ndarray],
            next_state: Dict[str, np.ndarray]):
        for k in state.keys():
            if k not in self._state_columns:
                raise ReplayBufferException(f'Column {k} it not visible.')
        for k in next_state.keys():
            if k not in self._state_columns:
                raise ReplayBufferException(f'Column {k} it not visible.')
        self._replay_buffer.add_transition(state, transition, next_state)

    def finish_current_sequence(self):
        self._replay_buffer.finish_current_sequence()


# class ReplayBufferTransitionListener(TransitionListener):
#     def __init__(self, replay_buffer: ReplayBuffer):
#         self.replay_buffer = replay_buffer

#     def update_transition(self, s, a, r, s_next, t, info):
#         self.replay_buffer.add_transition(
#             state=s,
#             transition={
#                 ACTION: a,
#                 REWARD: r,
#                 TERM: t
#             },
#             next_state=s_next
#         )
#         if t:
#             self.replay_buffer.finish_current_sequence()

#     def on_simulation_timeout(self):
#         self.replay_buffer.finish_current_sequence()


class TransitionIterator(object):
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            start_state_columns: List[str] = None,
            transition_columns: List[str] = None,
            next_state_columns: List[str] = None,
            idxs: List[int] = None):
        self._buffer = replay_buffer
        if start_state_columns is None:
            start_state_columns = self._buffer.get_state_column_names()
        self._start_state_columns = start_state_columns
        if transition_columns is None:
            transition_columns = self._buffer.get_transition_column_names()
        self._transition_columns = transition_columns
        if next_state_columns is None:
            next_state_columns = self._buffer.get_state_column_names()
        self._next_state_columns = next_state_columns
        if idxs is None:
            idxs = list(range(replay_buffer.num_transitions()))
        self._idxs = idxs

    def _idxs_to_batch(self, idxs: List[int]) -> List[np.ndarray]:
        batch = []
        for c in self._start_state_columns:
            batch.append(self._buffer.get_start_state_column(c, idxs))
        for c in self._transition_columns:
            batch.append(self._buffer.get_column(c, idxs))
        for c in self._next_state_columns:
            batch.append(self._buffer.get_next_state_column(c, idxs))
        return batch

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class TransitionIteratorEpochs(TransitionIterator):
    def __init__(self, *params, batch_size=32, shuffle=True, **kwargs):
        super().__init__(*params, **kwargs)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._i = 0

    def __iter__(self):
        if self._shuffle:
            random.shuffle(self._idxs)
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self._idxs):
            raise StopIteration
        idxs_batch = self._idxs[self._i:self._i + self._batch_size]
        self._i += self._batch_size
        return self._idxs_to_batch(idxs_batch)


class TransitionIteratorSampled(TransitionIterator):
    def __init__(self, *params, batch_size=32, num_samples=1000, **kwargs):
        super().__init__(*params, **kwargs)
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._cnt = 0

    def __iter__(self):
        self._cnt = 0
        return self

    def __next__(self):
        if self._cnt >= self._num_samples:
            raise StopIteration
        self._cnt += 1
        idxs_batch = []
        for _ in range(self._batch_size):
            idxs_batch.append(random.randint(0, len(self._idxs) - 1))
        return self._idxs_to_batch(idxs_batch)


class StateIterator(object):
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            state_columns: List[str] = None,
            idxs: List[int] = None):
        self._replay_buffer = replay_buffer
        if state_columns is None:
            state_columns = self._replay_buffer.get_state_column_names()
        self._state_columns = state_columns
        if idxs is None:
            idxs = list(range(replay_buffer.num_states()))
        self._idxs = idxs

    def _idxs_to_batch(self, idxs: List[int]) -> List[np.ndarray]:
        batch = []
        for c in self._state_columns:
            batch.append(self._replay_buffer.get_state_column(c, idxs))
        return batch

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class StateIteratorEpochs(StateIterator):
    def __init__(self, *params, batch_size=32, shuffle=False, **kwargs):
        super().__init__(*params, **kwargs)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._i = 0

    def __iter__(self):
        if self._shuffle:
            random.shuffle(self._idxs)
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self._idxs):
            raise StopIteration
        idxs_batch = self._idxs[self._i:self._i + self._batch_size]
        self._i += self._batch_size
        return self._idxs_to_batch(idxs_batch)


class StateIteratorSampled(StateIterator):
    def __init__(self, *params, batch_size=32, num_samples=1000, **kwargs):
        super().__init__(*params, **kwargs)
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._cnt = 0

    def __iter__(self):
        self._cnt = 0
        return self

    def __next__(self):
        if self._cnt >= self._num_samples:
            raise StopIteration
        self._cnt += 1
        idxs_batch = []
        for _ in range(self._batch_size):
            idxs_batch.append(random.randint(0, len(self._idxs) - 1))
        return self._idxs_to_batch(idxs_batch)
