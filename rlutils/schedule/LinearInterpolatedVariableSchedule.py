#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np

from .VariableSchedule import VariableSchedule
from typing import List, Union


class LinearInterpolatedVariableSchedule(VariableSchedule):
    """
    LinearInterpolatedVariableSchedule implements a variable schedule that 
    interpolates values linearly between boundary points that are passed into 
    the constructor.
    """

    def __init__(self, t_list: List[int], v_list: List[Union[int, float]]):
        """Piecewise Linear interpolated schedule. The value of the schedule is 
        linearly interpolated between the start value of each interval and the 
        start value of the next interval.

        Args:
            t_list (List[int]): Interval boundaries.
            v_list (List[Union[int, float]]): Values at boundaries.
        """
        self._t_list = np.array(t_list)
        self._v_list = np.array(v_list)

    def __call__(self, t: int) -> Union[int, float]:
        if len(np.where(self._t_list > t)[0]) == 0:
            return self._v_list[-1]

        i_low = np.max(np.where(self._t_list <= t)[0])
        i_high = np.min(np.where(self._t_list > t)[0])

        t_low = self._t_list[i_low]
        t_high = self._t_list[i_high]
        v_low = self._v_list[i_low]
        v_high = self._v_list[i_high]

        return v_low * (1. - (t - t_low) / (t_high - t_low)) + v_high * (t - t_low) / (t_high - t_low)

    @property
    def t_list(self) -> List[int]:
        return self._t_list.tolist()

    @property
    def v_list(self) -> List[Union[int, float]]:
        return self._v_list.tolist()
