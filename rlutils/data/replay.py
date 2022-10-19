#
# Copyright (c) 2022 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from copy import deepcopy
from ..types import TransitionSpec, Column


class TrajectoryBufferException(Exception):
    pass


class Trajectory(object):
    def __init__(self, transition_spec: Optional[TransitionSpec] = None):
        self._state_columns_config: Dict[str, Column] = {}
        self._transition_columns_config: Dict[str, Column] = {}
        self._state_columns: Dict[str, List[Any]] = {}
        self._transition_columns: Dict[str, List[Any]] = {}

        self._is_complete = False

        if transition_spec is not None:
            for col in transition_spec.state_columns:
                self._state_columns_config[col.name] = col
                self._state_columns[col.name] = []
            for col in transition_spec.transition_columns:
                self._transition_columns_config[col.name] = col
                self._transition_columns[col.name] = []

    @property
    def state_columns(self) -> Dict[str, Column]:
        return deepcopy(self._state_columns_config)

    @property
    def transition_columns(self) -> Dict[str, Column]:
        return deepcopy(self._transition_columns_config)

    @property
    def num_transitions(self) -> int:
        column_names = list(self._transition_columns.keys())
        if len(column_names) == 0:
            return 0
        return len(self._transition_columns[column_names[0]])

    @property
    def num_states(self) -> int:
        column_names = list(self._state_columns.keys())
        if len(column_names) == 0:
            return 0
        return len(self._state_columns[column_names[0]])

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    def __len__(self) -> int:
        return self.num_transitions

    def __append_state(
        self, 
        state: Dict[str, Union[int, float, bool, np.ndarray]]
    ):
        for name, col in self._state_columns_config.items():
            if name not in state.keys():
                raise TrajectoryBufferException(
                    f"State is missing {name} key.")
            value = np.reshape(state[name], col.shape).astype(col.dtype)
            self._state_columns[name].append(value)

    def __append_transition(
        self, 
        transition: Dict[str, Union[int, float, bool, np.ndarray]]
    ):
        for name, col in self._transition_columns_config.items():
            if name not in transition.keys():
                raise TrajectoryBufferException(
                    f"State is missing {name} key.")
            value = np.reshape(transition[name], col.shape).astype(col.dtype)
            self._transition_columns[name].append(value)

    def add_transition(
            self,
            start_state: Dict[str, Union[int, float, bool, np.ndarray]],
            transition:  Dict[str, Union[int, float, bool, np.ndarray]],
            next_state:  Dict[str, Union[int, float, bool, np.ndarray]]):
        if self._is_complete:
            raise TrajectoryBufferException(
                "Cannot add transitions to completed trajectory."
            )
        if self.num_transitions == 0:
            self.__append_state(start_state)
        self.__append_transition(transition)
        self.__append_state(next_state)

    def finish_current_sequence(self):
        self._is_complete = True

    def get_transition_column(
            self,
            name: str,
            idxs: Optional[List[int]] = None) -> List[Any]:
        if idxs is None:
            col = self._transition_columns[name]
        else:
            col = [self._transition_columns[name][i] for i in idxs]
        return col

    def set_transition_column(
            self,
            name: str,
            values: Union[List[Any], np.ndarray],
            idxs: Optional[List[int]] = None):
        if idxs is None:
            self._transition_columns[name][:] = values
        else:
            for i, c in zip(idxs, values):
                self._transition_columns[name][i] = c

    def add_transition_column(
        self,
        column: Column,
        values: Optional[np.ndarray] = None
    ):
        transitions = self.num_transitions
        if transitions > 0 \
                and values is not None \
                and np.shape(values)[0] != self.num_transitions:
            raise TrajectoryBufferException(
                "Column value does not have the correct length."
            )
        self._transition_columns_config[column.name] = column
        self._transition_columns[column.name] = []
        if transitions > 0 and values is not None:
            self._transition_columns[column.name] = [v for v in values]

    def get_state_column(
        self,
        name: str,
        idxs: Optional[List[int]] = None
    ) -> List[Any]:
        if idxs is None:
            col = self._state_columns[name]
        else:
            col = [self._state_columns[name][i] for i in idxs]
        return col

    def set_state_column(
        self,
        name: str,
        values: Union[List[Any], np.ndarray],
        idxs: Optional[List[int]] = None
    ):
        if idxs is None:
            self._state_columns[name][:] = values
        else:
            for i, v in zip(idxs, values):
                self._state_columns[name][i] = v

    def add_state_column(
        self,
        column: Column,
        values: Optional[np.ndarray] = None
    ):
        transitions = self.num_transitions
        if transitions > 0 \
                and values is not None \
                and np.shape(values)[0] != self.num_transitions + 1:
            raise TrajectoryBufferException(
                "Column value does not have the correct length."
            )
        self._state_columns_config[column.name] = column
        self._state_columns[column.name] = []
        if transitions > 0 and values is not None:
            self._state_columns[column.name] = [v for v in values]

    def get_start_state_column(
        self,
        name: str,
        idxs: Optional[List[int]] = None
    ) -> List[Any]:
        if idxs is None:
            col = self._state_columns[name][:-1]
        else:
            col = [self._state_columns[name][i] for i in idxs]
        return col

    def set_start_state_column(
        self,
        name: str,
        values: Union[List[Any], np.ndarray],
        idxs: Optional[List[int]] = None
    ):
        if idxs is None:
            self._state_columns[name][:-1] = values
        else:
            for i, v in zip(idxs, values):
                self._state_columns[name][i] = v

    def get_next_state_column(
        self,
        name: str,
        idxs: Optional[List[int]] = None
    ) -> List[Any]:
        col = self._state_columns[name][1:]
        if idxs is not None:
            col = [col[i] for i in idxs]
        return col

    def set_next_state_column(
        self,
        name: str,
        values: Union[List[Any], np.ndarray],
        idxs: Optional[List[int]] = None
    ):
        if idxs is None:
            self._state_columns[name][1:] = values
        else:
            for i, v in zip(idxs, values):
                self._state_columns[name][i + 1] = v


class TrajectoryBuffer:
    def __init__(self, transition_spec: Optional[TransitionSpec] = None):
        self._trajectory_list = [Trajectory(transition_spec)]
        self._transitions_idxs: List[Tuple[int, int]] = []
        self._state_idxs: List[Tuple[int, int]] = []

    @property
    def state_columns(self) -> Dict[str, Column]:
        return self._trajectory_list[0].state_columns

    @property
    def transition_columns(self) -> Dict[str, Column]:
        return self._trajectory_list[0].transition_columns

    @property
    def num_transitions(self) -> int:
        return len(self._transitions_idxs)

    @property
    def num_states(self) -> int:
        return len(self._state_idxs)

    def add_transition(
        self,
        start_state: Dict[str, Union[int, float, bool, np.ndarray]],
        transition: Dict[str, Union[int, float, bool, np.ndarray]],
        next_state: Dict[str, Union[int, float, bool, np.ndarray]]
    ):
        traj_idx = len(self._trajectory_list) - 1
        traj_len = len(self._trajectory_list[-1])

        if traj_len == 0:
            self._state_idxs.append((traj_idx, 0))
        self._trajectory_list[-1].add_transition(
            start_state, transition, next_state
        )
        self._transitions_idxs.append((traj_idx, traj_len))
        self._state_idxs.append((traj_idx, traj_len + 1))

    def finish_current_sequence(self):
        self._trajectory_list[-1].finish_current_sequence()
        new_trajectory = Trajectory()
        for column in self.transition_columns.values():
            new_trajectory.add_transition_column(column)
        for column in self.state_columns.values():
            new_trajectory.add_state_column(column)
        self._trajectory_list.append(new_trajectory)

    def _remap_transition_idxs(
        self,
        idxs: Optional[List[int]] = None
    ) -> List[Tuple[int, int]]:
        if idxs is None:
            return self._transitions_idxs
        return [self._transitions_idxs[i] for i in idxs]

    def _remap_state_idxs(
        self,
        idxs: Optional[List[int]] = None
    ) -> List[Tuple[int, int]]:
        if idxs is None:
            return self._state_idxs
        return [self._state_idxs[i] for i in idxs]

    def get_transition_column(
        self,
        name: str,
        idxs:  Optional[List[int]] = None
    ) -> np.ndarray:
        result = []
        for traj_idx, step_idx in self._remap_transition_idxs(idxs):
            traj = self._trajectory_list[traj_idx]
            result.append(traj.get_transition_column(name, [step_idx])[0])
        return np.copy(result)

    def set_transition_column(
        self,
        name: str,
        values: np.ndarray,
        idxs: Optional[List[int]] = None
    ):
        idxs_remapped = self._remap_transition_idxs(idxs)
        for (traj_idx, step_idx), value in zip(idxs_remapped, values):
            traj = self._trajectory_list[traj_idx]
            traj.set_transition_column(name, [value], [step_idx])

    def add_transition_column(
        self,
        column: Column,
        values: Optional[np.ndarray] = None
    ):
        for traj in self._trajectory_list:
            num_transitions = traj.num_transitions
            if values is None:
                traj.add_transition_column(column)
            else:
                traj.add_transition_column(column, values[:num_transitions])
                values = values[num_transitions:]

    def get_state_column(
        self,
        name: str,
        idxs: Optional[List[int]] = None
    ) -> np.ndarray:
        result = []
        for traj_idx, step_idx in self._remap_state_idxs(idxs):
            traj = self._trajectory_list[traj_idx]
            result.append(traj.get_state_column(name, [step_idx])[0])
        return np.copy(result)

    def set_state_column(
        self,
        name: str,
        values: np.ndarray,
        idxs: Optional[List[int]] = None
    ):
        idxs_remapped = self._remap_state_idxs(idxs)
        for (traj_idx, step_idx), value in zip(idxs_remapped, values):
            traj = self._trajectory_list[traj_idx]
            traj.set_state_column(name, [value], [step_idx])

    def add_state_column(
        self,
        column: Column,
        values: Optional[np.ndarray] = None
    ):
        for traj in self._trajectory_list:
            num_states = traj.num_states
            if values is None:
                traj.add_state_column(column)
            else:
                traj.add_state_column(column, values[:num_states])
                values = values[num_states:]

    def get_start_state_column(
        self,
        name: str,
        idxs:  Optional[List[int]] = None
    ) -> np.ndarray:
        result = []
        for traj_idx, step_idx in self._remap_transition_idxs(idxs):
            traj = self._trajectory_list[traj_idx]
            result.append(traj.get_start_state_column(name, [step_idx])[0])
        return np.copy(result)

    def set_start_state_column(
        self,
        name: str,
        values: np.ndarray,
        idxs: Optional[List[int]] = None
    ):
        idxs_remapped = self._remap_transition_idxs(idxs)
        for (traj_idx, step_idx), value in zip(idxs_remapped, values):
            traj = self._trajectory_list[traj_idx]
            traj.set_start_state_column(name, [value], [step_idx])

    def get_next_state_column(
        self,
        name: str,
        idxs: Optional[List[int]] = None
    ) -> np.ndarray:
        result = []
        for traj_idx, step_idx in self._remap_transition_idxs(idxs):
            traj = self._trajectory_list[traj_idx]
            result.append(traj.get_next_state_column(name, [step_idx])[0])
        return np.copy(result)

    def set_next_state_column(
        self,
        name: str,
        values: np.ndarray,
        idxs: Optional[List[int]] = None
    ):
        idxs_remapped = self._remap_transition_idxs(idxs)
        for (traj_idx, step_idx), value in zip(idxs_remapped, values):
            traj = self._trajectory_list[traj_idx]
            traj.set_next_state_column(name, [value], [step_idx])

    # def get_transition(
    #         self,
    #         start_state_columns: List[str],
    #         transition_columns: List[str],
    #         next_state_columns: List[str]) -> Tuple[np.ndarray]:


class TrajectoryBufferFixedSize(TrajectoryBuffer):
    def __init__(
        self,
        max_transitions: int,
        transition_spec: Optional[TransitionSpec] = None
    ):
        super(TrajectoryBufferFixedSize, self).__init__(transition_spec)
        self._max_transitions = max_transitions

    @property
    def max_transitions(self) -> int:
        return self._max_transitions

    def _trim_buffer(self):
        if len(self._transitions_idxs) > self._max_transitions:
            self._transitions_idxs.pop(0)
            self._state_idxs.pop(0)

            """
            Check if the last transition of first trajectory was removed.
            If so, remove the end state from _state_idxs, remove the first 
            trajectory, and re-index _transitions_idxs and _state_idxs.
            """
            if self._transitions_idxs[0][0] == 1:
                self._state_idxs.pop(0)
                self._trajectory_list.pop(0)
                for i in range(len(self._transitions_idxs)):
                    traj_idx = self._transitions_idxs[i][0]
                    step_idx = self._transitions_idxs[i][1]
                    self._transitions_idxs[i] = (traj_idx - 1, step_idx)
                for i in range(len(self._state_idxs)):
                    traj_idx = self._state_idxs[i][0]
                    step_idx = self._state_idxs[i][1]
                    self._state_idxs[i] = (traj_idx - 1, step_idx)

    def add_transition(
        self,
        start_state: Dict[str, Union[int, float, bool, np.ndarray]],
        transition: Dict[str, Union[int, float, bool, np.ndarray]],
        next_state: Dict[str, Union[int, float, bool, np.ndarray]]
    ):
        super(TrajectoryBufferFixedSize, self).add_transition(
            start_state, transition, next_state
        )
        self._trim_buffer()
