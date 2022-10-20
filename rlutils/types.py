from abc import abstractmethod
from typing import List, Dict, Any
from collections import namedtuple
import numpy as np


Action = "action"
Reward = "reward"
Term = "term"

Column = namedtuple("Column", ["name", "shape", "dtype"])

action_index_column = Column(Action, shape=(), dtype=int)
reward_column = Column(Reward, shape=(), dtype=float)
term_column = Column(Term, shape=(), dtype=bool)


class TransitionSpec:
    def __init__(
            self,
            state_columns: List[Column],
            transition_columns: List[Column]):
        self._state_columns = state_columns
        self._transition_columns = transition_columns

    @property
    def state_columns(self) -> List[Column]:
        return self._state_columns

    @property
    def transition_columns(self) -> List[Column]:
        return self._transition_columns


class TransitionListener:
    @abstractmethod
    def update_transition(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        reward: Any, 
        next_state: Dict[str, Any], 
        term: bool, 
        info: Dict[Any, Any]
    ): # pragma: no cover
        """
        Update agent with a transition. Term is a flag that terminates the 
        interaction with the given task.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param term:
        """
        pass

    @abstractmethod
    def on_simulation_timeout(self): # pragma: no cover
        """
        This method is called when a SimulationTimout exception is raised while 
        simulating a trajectory. This occurs every time a trajectory finished in
        a state different than a terminal state.
        """
        pass


class Agent(TransitionListener):
    """
    Super class for all value-based agents.
    """

    @abstractmethod
    def q_values(self, state: Dict[str, Any]) -> np.ndarray:  # pragma: no cover
        """
        Q-values for a given state.

        :param state: State of an MDP.
        :return: A vector of dimension [num_actions]. Each entry contains the 
            Q-values for each action at the provided state.
        """

    @abstractmethod
    def reset(self, *params, **kwargs):  # pragma: no cover
        """
        Reset agent to its initialization.
        :return:
        """


class Policy:
    '''
    Base class for all policies.
    '''

    @abstractmethod
    def __call__(self, state: Dict[str, Any]) -> Any:  # pragma: no cover
        """Select an action at the given state.

        Args:
            state (Dict[str, Any]): State

        Returns:
            Any: Action
        """
        pass


class _TransitionListenerAggregator(TransitionListener):
    def __init__(self, *update_listener_list: TransitionListener):
        self._update_listener_list = tuple(update_listener_list)

    def update_transition(self, *params, **kvargs):
        for l in self._update_listener_list:
            l.update_transition(*params, **kvargs)

    def on_simulation_timeout(self):
        for l in self._update_listener_list:
            l.on_simulation_timeout()


def transition_listener(*update_listener_list: TransitionListener):
    return _TransitionListenerAggregator(*update_listener_list)
