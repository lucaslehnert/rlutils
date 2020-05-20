#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from abc import ABC, abstractmethod


def simulate(mdp, policy, transition_listener, max_steps=5000):
    """
    Train the agent on the given MDP for one episode.

    Episodes can terminate in two ways, either a terminal state is reached or the max_step count is reached. If a
    terminal state is reached, then the info dictionary contains the key-pair
        terminate_reason: terminal_state
    If the episode is terminated due to the step count being reached, then the info dictionary contains the key-pair
        terminate_reason: simulation_timeout. In this case, a rlutils.data.SimulationTimout exception is also thrown.
        Please refer to rlutils.data.simulate_gracefully which does not throw a SimulationTimout exception but has the
        same behaviour otherwise.
    If the episode does not terminate, then the key 'terminate_reason' will not be added to the info dictionary.

    :param mdp: MDP
    :param policy: Policy
    :param update_callback: Update callback accepting (s, a, r, s_next, t, info).
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
            raise SimulationTimout('Timeout of %d steps hit.' % max_steps)


def simulate_gracefully(mdp, policy, transition_listener, max_steps=5000):
    """
    Train the agent on the given MDP for one episode. This function cannot raise a SimulationTimout exception and
    returns instead. No terminal flag is fed to the agent.

    :param mdp: MDP
    :param policy: Policy
    :param update_callback: Update callback accepting (s, a, r, s_next, t, info).
    :return: None
    """
    try:
        simulate(mdp, policy, transition_listener, max_steps=max_steps)
    except SimulationTimout:
        pass


def replay_trajectory(trajectory, transition_listener):
    """
    Replay the given trajectory through all a transition listener.

    :param trajectory: Trajectory that is replayed.
    :param transition_listener: Transition listener that is updated.
    :return: None

    The i variable can be used for debugging and counting the different transitions. Otherwise it has not other
    function.
    """
    for i, (s, a, r, s_next, done, info) in enumerate(zip(*trajectory.all())):
        transition_listener.update_transition(s, a, r, s_next, done, info)



class SimulationTimout(Exception):
    pass


class TransitionListener(ABC):
    @abstractmethod
    def update_transition(self, s, a, r, s_next, t, info): # pragma: no cover
        """
        Update agent with a transition. Term is a flag that terminates the interaction with the given task.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param term:
        :return: None or a dictionary with learning statistics or info about the agent.
        """
        pass


class _TransitionListenerAggregator(TransitionListener):
    def __init__(self, *update_listener_list):
        self._update_listener_list = update_listener_list

    def update_transition(self, s, a, r, s_next, t, info):
        for l in self._update_listener_list:
            l.update_transition(s, a, r, s_next, t, info)


def transition_listener(*update_listener_list):
    return _TransitionListenerAggregator(*update_listener_list)
