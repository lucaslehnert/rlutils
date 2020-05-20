#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from abc import abstractmethod


class Policy(object):
    '''
    Base class for all policies.
    '''

    @abstractmethod
    def __call__(self, state):  # pragma: no cover
        '''
        Select an action at the given state.

        :param state: A state.
        :return: The action the policy selects.
        '''
        pass
