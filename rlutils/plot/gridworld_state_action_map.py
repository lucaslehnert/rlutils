#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
import rlutils as rl
import plotly
import plotly.graph_objects as go
from itertools import product

from .ColorMap import ColorMap
from .utils import get_shape_up, get_shape_right, get_shape_down, get_shape_left


def _get_arrow_shapes(x, y, q_vals, color_map):  # pragma: no cover
    return [
        get_shape_up(x, y, color_map(q_vals[rl.environment.gridworld.GridWorldAction.UP])),
        get_shape_right(x, y, color_map(q_vals[rl.environment.gridworld.GridWorldAction.RIGHT])),
        get_shape_down(x, y, color_map(q_vals[rl.environment.gridworld.GridWorldAction.DOWN])),
        get_shape_left(x, y, color_map(q_vals[rl.environment.gridworld.GridWorldAction.LEFT]))
    ]


def gridworld_state_action_map(q_values, c_range=None, color_scale=plotly.colors.sequential.Blues):  # pragma: no cover
    if c_range is None:
        c_range = [np.min(q_values), np.max(q_values)]

    color_map = ColorMap(color_scale, c_range)
    _, num_x, num_y = np.shape(q_values)

    x_pos, y_pos = np.meshgrid(range(num_x), range(num_y))
    x_pos = np.reshape(x_pos, -1)
    y_pos = np.reshape(y_pos, -1)
    q_values_lin = np.stack([np.reshape(q_values[a].transpose(), -1) for a in range(4)])

    x_scatter = np.concatenate((x_pos, x_pos + .25, x_pos, x_pos - .25))
    y_scatter = np.concatenate((y_pos - .25, y_pos, y_pos + .25, y_pos))
    v_scatter = np.concatenate((q_values_lin[0], q_values_lin[1], q_values_lin[2], q_values_lin[3]))
    v_scatter = ['{:1.5e}'.format(v) for v in v_scatter]

    data = [go.Scatter(
        x=x_scatter,
        y=y_scatter,
        text=v_scatter,
        mode='markers',
        marker=dict(
            size=0,
            colorscale=plotly.colors.sequential.Blues,
            showscale=True,
            cmin=color_map.get_min(),
            cmax=color_map.get_max(),
            colorbar=dict(thickness=10, tickvals=np.linspace(color_map.get_min(), color_map.get_max(), 6)),
        )
    )]

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

    arrow_shapes = []
    for x, y in product(range(num_x), range(num_y)):
        arrow_shapes += _get_arrow_shapes(x, y, q_values[:, x, y], color_map)
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
