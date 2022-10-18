#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from abc import abstractmethod
from typing import Dict, Any, List
import gym
from ..policy import Policy



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


class _TransitionListenerAggregator(TransitionListener):
    def __init__(self, *update_listener_list: TransitionListener...):
        self._update_listener_list = tuple(update_listener_list)

    def update_transition(self, *params, **kvargs):
        for l in self._update_listener_list:
            l.update_transition(*params, **kvargs)

    def on_simulation_timeout(self):
        for l in self._update_listener_list:
            l.on_simulation_timeout()


def transition_listener(*update_listener_list: TransitionListener...):
    return _TransitionListenerAggregator(*update_listener_list)



def simulate(
    mdp: gym.Env, 
    policy: Policy, 
    transition_listener: TransitionListener, 
    max_steps: int=5000
):
    """
    Train the agent on the given MDP for one episode.

    Episodes can terminate in two ways, either a terminal state is reached or 
    the max_step count is reached. If a terminal state is reached, then the info 
    dictionary contains the key-pair terminate_reason: terminal_state. If the 
    episode is terminated due to the step count being reached, then the info 
    dictionary contains the key-pair terminate_reason: simulation_timeout. In 
    this case, a rlutils.data.SimulationTimout exception is also thrown.
    Please refer to rlutils.data.simulate_gracefully which does not throw a 
    SimulationTimout exception but has the
    same behaviour otherwise.
    If the episode does not terminate, then the key 'terminate_reason' will not 
    be added to the info dictionary.

    :param mdp: MDP
    :param policy: Policy
    :param transition_listener: Object implementing TransitionListener
    :return: None
    """
    s = mdp.reset()
    done = False
    i = 0
    while not done:
        a = policy(s)
        s_next, r, done, info = mdp.step(a)

        if done:
            info['terminate_reason'] = 'terminal_state'
        elif i + 1 >= max_steps:
            info['terminate_reason'] = 'simulation_timeout'

        transition_listener.update_transition(s, a, r, s_next, done, info)
        s = s_next

        i += 1
        if i >= max_steps and not done:
            transition_listener.on_simulation_timeout()
            break
