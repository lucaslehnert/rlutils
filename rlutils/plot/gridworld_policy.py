#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
import rlutils as rl
import plotly.graph_objects as go
from itertools import product

from .utils import get_shape_up, get_shape_right, get_shape_down, get_shape_left


def _get_action_arrow(x, y, action):  # pragma: no cover
    if action == rl.environment.gridworld.GridWorldAction.UP:
        return get_shape_up(x, y, 'black')
    elif action == rl.environment.gridworld.GridWorldAction.RIGHT:
        return get_shape_right(x, y, 'black')
    elif action == rl.environment.gridworld.GridWorldAction.DOWN:
        return get_shape_down(x, y, 'black')
    elif action == rl.environment.gridworld.GridWorldAction.LEFT:
        return get_shape_left(x, y, 'black')


def gridworld_policy(action_matrix):  # pragma: no cover
    num_x, num_y = np.shape(action_matrix)
    data = None

    grid_lines = []
    for i in range(num_x + 1):
        grid_lines.append(
            go.layout.Shape(
                type="line",
                xref="x",
                yref="y",
                x0=i - .5,
                y0=-.5,
                x1=i - .5,
                y1=9.5,
                line=dict(
                    color="LightGray",
                    width=1,
                )
            )
        )
    for i in range(num_y + 1):
        grid_lines.append(
            go.layout.Shape(
                type="line",
                xref="x",
                yref="y",
                x0=-.5,
                y0=i - .5,
                x1=9.5,
                y1=i - .5,
                line=dict(
                    color="LightGray",
                    width=1,
                )
            )
        )
    arrow_shapes = [_get_action_arrow(x, y, action_matrix[x, y]) for x, y in product(range(num_x), range(num_y))]
    layout = dict(
        width=500,
        height=500,
        xaxis=dict(
            range=[-.55, num_x - .45],
            tick0=0,
            dtick=1.0,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            range=[num_y - .45, -.55],
            tick0=0,
            dtick=1.0,
            showgrid=True,
            zeroline=False
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        shapes=grid_lines + arrow_shapes
    )

    return data, layout
