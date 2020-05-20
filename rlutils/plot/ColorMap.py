#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
import colorlover as cl


class ColorMap(object):  # pragma: no cover
    def __init__(self, color_scale, color_range, num_interp=500):  # pragma: no cover
        self._color_list = cl.interp(color_scale, num_interp)
        self._color_val_bounds = np.linspace(color_range[0], color_range[1], num_interp)

    def get_min(self):  # pragma: no cover
        return self._color_val_bounds[0]

    def get_max(self):  # pragma: no cover
        return self._color_val_bounds[-1]

    def __call__(self, v):  # pragma: no cover
        i = np.max(np.where(self._color_val_bounds <= v)[0])
        return self._color_list[i]
