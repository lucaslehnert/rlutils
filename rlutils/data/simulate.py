#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import gym
from ..types import Policy, TransitionListener


def simulate(
    mdp: gym.Env,
    policy: Policy,
    transition_listener: TransitionListener
):
    """Simulate an agent on the given MDP for one episode.

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

    Args:
        mdp (gym.Env): Task.
        policy (Policy): Policy used for action selection.
        transition_listener (TransitionListener): Transition listener.
    """
    s, _ = mdp.reset()
    done = False
    trunc = False
    while not done and not trunc:
        a = policy(s)
        s_next, r, done, trunc, info = mdp.step(a)

        transition_listener.update_transition(s, a, r, s_next, done, info)
        s = s_next

        if not done and trunc:
            transition_listener.on_simulation_timeout()
