#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import plotly.graph_objects as go


def get_shape_up(x, y, color):  # pragma: no cover
    path_fmt_str = 'M {:f} {:f} L {:f} {:f} L {:f} {:f} L {:f} {:f} Z'
    return go.layout.Shape(
        type="path",
        path=path_fmt_str.format(x, y - .45, x - .15, y - .15, x, y, x + .15, y - .15),
        fillcolor=color,
        line=dict(width=0)
    )


def get_shape_right(x, y, color):  # pragma: no cover
    path_fmt_str = 'M {:f} {:f} L {:f} {:f} L {:f} {:f} L {:f} {:f} Z'
    return go.layout.Shape(
        type="path",
        path=path_fmt_str.format(x + .45, y, x + .15, y + .15, x, y, x + .15, y - .15),
        fillcolor=color,
        line=dict(width=0)
    )


def get_shape_down(x, y, color):  # pragma: no cover
    path_fmt_str = 'M {:f} {:f} L {:f} {:f} L {:f} {:f} L {:f} {:f} Z'
    return go.layout.Shape(
        type="path",
        path=path_fmt_str.format(x, y + .45, x + .15, y + .15, x, y, x - .15, y + .15),
        fillcolor=color,
        line=dict(width=0)
    )


def get_shape_left(x, y, color):  # pragma: no cover
    path_fmt_str = 'M {:f} {:f} L {:f} {:f} L {:f} {:f} L {:f} {:f} Z'
    return go.layout.Shape(
        type="path",
        path=path_fmt_str.format(x - .45, y, x - .15, y + .15, x, y, x - .15, y - .15),
        fillcolor=color,
        line=dict(width=0)
    )
