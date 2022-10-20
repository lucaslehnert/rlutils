import gym
from abc import abstractmethod
from ..types import TransitionSpec
from typing import Any, Tuple, Dict


class Task(gym.Env):
    @property
    @abstractmethod
    def transition_spec(self) -> TransitionSpec:
        """Returns a transition specification object.

        Returns:
            TransitionConfig: Transition specification for a task. 
        """
        pass
    
    @abstractmethod
    def step(
        self,
        action: Any
    ) -> Tuple[Dict[str, Any], Any, bool, bool, dict]:
        pass
    
    @abstractmethod
    def reset(self, *args: Any, **kvargs: Any) -> Tuple[Dict[str, Any], dict]:
        pass

