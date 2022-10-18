#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from abc import abstractmethod
from typing import Dict, Any


class Policy(object):
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
