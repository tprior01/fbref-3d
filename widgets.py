from dash import dcc
from dash import html


def drop_down(id, options, value, style):
    return html.Div([
        dcc.Dropdown(
            id=id,
            options=options if isinstance(options, list) else [{"label": k, "value": v} for k, v in options.items()],
            value=value,
            clearable=False
        ),
    ],
        style=style
    )


def check_list(id, options):
    return dcc.Checklist(
        options=options if isinstance(options, list) else [{'label': k, 'value': v} for k, v in options.items()],
        value=options if isinstance(options, list) else [value for value in options.values()],
        inline=True,
        id=id
    )


def radio_item(id, options, value, style):
    return dcc.RadioItems(
        options=options if isinstance(options, list) else [{'label': k, 'value': v} for k, v in options.items()],
        value=value,
        id=id,
        labelStyle=style
    )


def range_slider(id, min, max, value_min, value_max, step):
    return dcc.RangeSlider(
        min=min,
        max=max,
        value=[value_min, value_max],
        step=step,
        allowCross=False,
        id=id
    )
