#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np


def mean_with_max_min(xvals, yvals, axis=0, color='C0', label=None):  # pragma: no cover
    import matplotlib.pyplot as plt # Must be imported on use, otherwise matplotlib may open a UI window.

    yvals_m = np.mean(yvals, axis=axis)
    yvals_max = np.max(yvals, axis=axis)
    yvals_min = np.min(yvals, axis=axis)
    plt.plot(xvals, yvals_m, c=color, label=label)
    plt.fill_between(xvals, y1=yvals_m + yvals_max, y2=yvals_m - yvals_min, color=color, alpha=0.2)
