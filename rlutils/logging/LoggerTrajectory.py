#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#


from ..types import TransitionListener, action_index_column, reward_column, term_column
from ..data import  TrajectoryBuffer
from typing import Dict, Any, Optional


class LoggerTrajectory(TransitionListener):
    def __init__(
        self, 
        buffer: TrajectoryBuffer,
        action_column_name: Optional[str] = None,
        reward_column_name: Optional[str] = None,
        term_column_name: Optional[str] = None,
    ):
        if action_column_name is None:
            action_column_name = action_index_column.name
        if reward_column_name is None:
            reward_column_name = reward_column.name
        if term_column_name is None:
            term_column_name = term_column.name
        self._buffer = buffer
        self._action_column_name = action_column_name
        self._reward_column_name = reward_column_name
        self._term_column_name = term_column_name

    def update_transition(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        reward: Any, 
        next_state: Dict[str, Any], 
        term: bool, 
        _: Dict[Any, Any]
    ):
        self._buffer.add_transition(
            start_state=state,
            transition={
                self._action_column_name: action,
                self._reward_column_name: reward,
                self._term_column_name: term
            },
            next_state=next_state
        )
        if term:
            self._buffer.finish_current_sequence()

    def on_simulation_timeout(self):
        self._buffer.finish_current_sequence()
