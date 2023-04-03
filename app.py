from widgets import radio_item, drop_down, check_list, range_slider
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from plotly.express import scatter_3d, scatter
from maps import cat_options, sub_options, seasons, comps, positions, not_per_min, aggregates
from pandas import read_sql
from os import environ
from dotenv import load_dotenv
from sqlalchemy import text, create_engine, select, MetaData, func, extract, Integer, true, cast

load_dotenv()

engine = create_engine(environ["SQLALCHEMY_DATABASE_URI"])

metadata = MetaData()
metadata.reflect(engine, only=['player', 'keepers', 'keepersadv', 'shooting', 'passing', 'passing_types',
                               'gca', 'defense', 'possession', 'playingtime', 'misc'])


app = dash.Dash()

button_style = {'width': '20%', 'display': 'inline-block'}
radio_style = {'display': 'inline-block', 'marginTop': '5px'}

with engine.connect() as conn:
    max_mins = conn.execute(text("""select max(sum) from (select id, sum(minutes) from playingtime group by (id)) as x""")).scalar()
    max_value, max_age = conn.execute(text("""select ceiling(max(current_value))::Integer, max(date_part('year', age(dob)))::Integer from player""")).all()[0]
    conn.close()

app.layout = html.Div([
    html.Div([
        html.Label(['x'], style={'font-weight': 'bold', "text-align": "center"}),
        html.Div([
            drop_down("xcat", cat_options, "shooting", button_style),
            drop_down("x", sub_options["shooting"], "xg", button_style)
        ])
    ]),
    html.Div([
        drop_down("ycat", cat_options, "shooting", button_style),
        drop_down("y", sub_options["shooting"], "goals", button_style)
    ]),
    html.Div([
        drop_down("zcat", cat_options, "shooting", button_style),
        drop_down("z", sub_options["shooting"], "shot_on_target", button_style)
    ]),
    check_list("seasons", seasons),
    check_list("competitions", comps),
    check_list("positions", positions),

    radio_item(id="dimension", options={"2D": True, "3D": False}, value=True, style=radio_style),
    radio_item(id="colour", options={"Z-Axis": "z", "Position": "position"}, value="z", style=radio_style),
    radio_item(id="total", options={"Total": True, "Per 90": False}, value=True, style=radio_style),

    range_slider("ages", 15, max_age, 15, max_age, 1),
    range_slider("values", 0, max_value, 0, max_value, 1),
    range_slider("minutes", 0, max_mins, 1250, max_mins, 100),

    drop_down("nation", ["All"] + [], "All", button_style),
    drop_down("club", ["All"] + [], "All", button_style),

    html.Div([dcc.Graph(id='main-plot', config={'displayModeBar': False})])
])


@app.callback(
    Output('x', 'options'),
    Input('xcat', 'value'),
    prevent_initial_call=True
)
def update_x_dropdown(value):
    return [{"label": key, "value": value} for key, value in sub_options[value].items()]


@app.callback(
    Output('x', 'value'),
    Input('x', 'options'),
    prevent_initial_call=True
)
def update_x_value(options):
    return options[0]['value']


@app.callback(
    Output('y', 'options'),
    Input('ycat', 'value'),
    prevent_initial_call=True
)
def update_y_dropdown(value):
    return [{"label": key, "value": value} for key, value in sub_options[value].items()]


@app.callback(
    Output('y', 'value'),
    Input('y', 'options'),
    prevent_initial_call=True
)
def update_y_value(options):
    return options[0]['value']


@app.callback(
    Output('z', 'options'),
    Input('zcat', 'value'),
    prevent_initial_call=True
)
def update_z_dropdown(value):
    return [{"label": key, "value": value} for key, value in sub_options[value].items()]


@app.callback(
    Output('z', 'value'),
    Input('z', 'options'),
    prevent_initial_call=True

)
def update_z_value(options):
    return options[0]['value']


@app.callback(
    output=Output('main-plot', 'figure'),
    inputs=[
        (
            Input('x', 'value'),
            Input('y', 'value'),
            Input('z', 'value'),
            State('x', 'options'),
            State('y', 'options'),
            State('z', 'options'),

        ),
        (
            Input('xcat', 'value'),
            Input('ycat', 'value'),
            Input('zcat', 'value'),
        ),
        Input('club', 'value'),
        Input('nation', 'value'),
        Input('ages', 'value'),
        Input('values', 'value'),
        Input('minutes', 'value'),
        Input('seasons', 'value'),
        Input('competitions', 'value'),
        Input('positions', 'value'),
        Input('colour', 'value'),
        Input('dimension', 'value'),
        Input('total', 'value'),

    ],
    prevent_initial_call=True
)
def update(xyz, xyz_cats, club, nationality, ages, values, minutes, seasons, comps, positions, colour, dim, total):
    query = make_query(xyz, xyz_cats, club, nationality, ages, values, minutes, seasons, comps, positions, total)
    with engine.connect() as conn:
        graph_params = dict(
            data_frame=read_sql(query, con=conn),
            x='x',
            y='y',
            color=colour,
            hover_data=[
                'name', 'name',
                'nationality', 'nationality',
            ],
            labels={
                "x": [x['label'] for x in xyz[3] if x['value'] == xyz[0]][0],
                "y": [x['label'] for x in xyz[4] if x['value'] == xyz[1]][0],
                "z": [x['label'] for x in xyz[5] if x['value'] == xyz[2]][0],
                "name": "name",
                'nation': "nationality",
                'position': "position"
            },
        )
    if dim:
        plot = scatter
    else:
        plot = scatter_3d
        graph_params['z'] = 'z'

    fig = plot(**graph_params)
    fig_updates(fig)
    conn.close()
    return fig


def fig_updates(fig):
    fig.update_layout(
        width=900,
        height=600,
        autosize=False,
        margin=dict(t=0, b=0, l=0, r=0),
        template="plotly_white",
    )
    fig.update_traces(marker_size=5)
    fig.update_scenes(
        aspectratio=dict(x=0.9, y=0.9, z=0.7),
        aspectmode="manual"
    )


def select_clause(sub, table):
    if sub in aggregates.keys():
        v = aggregates[sub]
        return cast(func.sum(table.c[v[0]]) / (1.0 * func.coalesce(func.nullif(func.sum(table.c[v[1]]), 0), 1)) * v[2], v[3])
    else:
        return func.sum(table.c[sub])


def data_sub_query(sub, cat, seasons, comps, label):
    table = metadata.tables[str(cat)]
    query = select(
        table.c["id"],
        select_clause(sub, table).label(label)
    ).group_by(
        table.c["id"]
    ).where(
        table.c["season"].in_(seasons),
        table.c["comp"].in_(comps)
    ).subquery()
    return query


def player_sub_query(club, nationality, values, positions):
    player = metadata.tables["player"]
    query = select(
        player.c["id"],
        player.c["name"],
        player.c["club"],
        player.c["nationality"],
        player.c["position"],
        player.c["current_value"],
        extract('year', func.age(player.c["dob"])).label('age').cast(Integer),
    ).where(
        (player.c["club"] == club if club != 'All' else true()) &
        (player.c["nationality"] == nationality if nationality != 'All' else true()) &
        (player.c["current_value"].between(values[0], values[1])) &
        (player.c["position"].in_(positions))
    ).subquery()
    return query


def mins_sub_query(seasons, comps, minutes):
    table = metadata.tables["playingtime"]
    query = select(
        table.c["id"],
        func.sum(table.c["minutes"]).label('mins')
    ).group_by(
        table.c["id"],
    ).where(
        table.c["season"].in_(seasons),
        table.c["comp"].in_(comps)
    ).where(
        table.c["minutes"].between(minutes[0], minutes[1])
    ).subquery()
    return query


def make_query(xyz, xyz_cats, club, nationality, ages, values, minutes, seasons, comps, positions, total):
    player = player_sub_query(club, nationality, values, positions)
    x = data_sub_query(xyz[0], xyz_cats[0], seasons, comps, 'x')
    y = data_sub_query(xyz[1], xyz_cats[1], seasons, comps, 'y')
    z = data_sub_query(xyz[2], xyz_cats[2], seasons, comps, 'z')
    mins = mins_sub_query(seasons, comps, minutes)

    query = select(
        player.c["id"],
        player.c["name"],
        player.c["club"],
        player.c["nationality"],
        player.c["position"],
        player.c["current_value"],
        player.c["age"],
        x.c["x"] if total and xyz[0] in not_per_min else (x.c["x"] / mins.c["mins"] * 90).label('x'),
        y.c["y"] if total and xyz[1] in not_per_min else (y.c["y"] / mins.c["mins"] * 90).label('y'),
        z.c["z"] if total and xyz[2] in not_per_min else (z.c["z"] / mins.c["mins"] * 90).label('z')
    ).join(
        x,
        x.c["id"] == player.c["id"],
        isouter=True
    ).join(
        y,
        y.c["id"] == player.c["id"],
        isouter=True
    ).join(
        z,
        z.c["id"] == player.c["id"],
        isouter=True
    ).join(
        mins,
        mins.c["id"] == player.c["id"],
    ).where(
        player.c["age"].between(ages[0], ages[1]),
    )
    return query


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
