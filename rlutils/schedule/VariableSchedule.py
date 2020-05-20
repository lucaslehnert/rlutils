#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

from abc import abstractmethod


class VariableSchedule(object):
    """
    Abstract super class for a schedule for a single scalar number. These schedules map a time step integer
    t = 0,1,2,... to some scalar value.
    """

    @abstractmethod
    def __call__(self, t):  # pragma: no cover
        """
        Return the variable value for the given time index value.

        :param t: The time index value.
        :return: Variable value.
        """
