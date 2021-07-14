import gym
import numpy as np
from abc import abstractmethod
from typing import Dict


class Task(gym.Env):
    @abstractmethod
    def state_defaults(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary of default state key-value paris. This 
        dictionary resembles an examplary state dictionary used to initialize
        replay buffers. The values in this dictionary can be set to zero or 
        another custom default value.

        :return: Default state dictionary.
        :rtype: Dict[str, np.ndarray]
        """
        pass

    @abstractmethod
    def transition_defaults(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary of default transition key-value paris. This 
        dictionary resembles an examplary transition dictionary used to 
        initialize replay buffers. The values in this dictionary can be set to 
        zero or another custom default value. 

        This dictionary must contain the key `rlutils.data.ACTION`, 
        `rlutils.data.REWARD`, and `rlutils.data.TERM`.

        :return: Default transition dictionary.
        :rtype: Dict[str, np.ndarray]
        """
        pass
