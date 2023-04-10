from widgets import radio_item, drop_down, check_list, range_slider
from dash import Dash
from dash_bootstrap_components import Col, Row, Card, themes
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State, ClientsideFunction
from plotly.express import scatter_3d, scatter
from maps import cat_options, axis_options, seasons, comps, positions, not_per_min, aggregates
from pandas import read_sql
from os import environ
from sqlalchemy import text, create_engine, select, MetaData, func, extract, Integer, true, cast

# from dotenv import load_dotenv
#
# load_dotenv()

engine = create_engine(environ["SQLALCHEMY_DATABASE_URI"])

metadata = MetaData()
metadata.reflect(engine, only=cat_options.values())

app = Dash(external_stylesheets=[themes.BOOTSTRAP])
server = app.server


with engine.connect() as conn:
    max_mins = conn.execute(
        text("select max(sum) from (select id, sum(minutes) from playingtime group by (id)) as x")).scalar()
    max_value, max_age = conn.execute(
        text("select ceiling(max(current_value))::Integer, max(date_part('year', age(dob)))::Integer from player")).all()[0]
    clubs = conn.execute(
        text("select distinct club from player where current_value > 20.0 order by club")).scalars().all()
    nations = conn.execute(
        text("select distinct nationality from player order by nationality")).scalars().all()
    names = conn.execute(
        text("select name from player order by name")).scalars().all()
    conn.close()

app.layout = Col([
    dcc.Store(id='selected-data'),
    dcc.Markdown('''
        # FBREF-3D
    
        A dashboard to visualise the data on fbref.com as 2D or 3D scatter graphs. The data is 
        periodically scraped and added to a PostgreSQL database. 
    '''),
    Card([
        Row([
            Col(html.Label('x-axis')),
            Col(html.Label('y-axis')),
            Col(html.Label('z-axis'))
        ]),
        Row([
            Col([
                Row([
                    drop_down("xcat", cat_options, "shooting"),
                    drop_down("x", axis_options["shooting"], "xg")
                ])
            ]),
            Col([
                Row([
                    drop_down("ycat", cat_options, "passing"),
                    drop_down("y", axis_options["passing"], "xa")
                ])
            ]),
            Col([
                Row([
                    drop_down("zcat", cat_options, "passing"),
                    drop_down("z", axis_options["passing"], None)
                ])
            ])
        ]),
    ], style={"width": "100%", "height": "50%"}, body=True),
    Card([
        Row([
            Col(html.Label('scatter type')),
            Col(html.Label('marker colours')),
            Col(html.Label('per minute or totals'))
        ]),
        Row([
            Col([Row([radio_item(id="dimension", options={"3D": True, "2D": False}, value=True)])]),
            Col([Row([radio_item(id="colour", options={"Z-Axis": "z", "Position": "position"}, value="z")])]),
            Col([Row([radio_item(id="per_min", options={"Per 90": True, "Total": False}, value=True)])])
        ]),
    ], style={"width": "100%"}, body=True),
    Card([
        Row([
            Col([
                Row(html.Label('seasons')),
                Row(check_list("seasons", seasons)),
            ]),
            Col([
                Row(html.Label('competitions')),
                Row(check_list("competitions", comps)),
            ]),
            Col([
                Row(html.Label('positions')),
                Row(check_list("positions", positions)),
            ])
        ])
    ], style={"width": "100%"}, body=True),

    Card([
        Row([
            Col(html.Label('age')),
            Col(html.Label('value (€)')),
            Col(html.Label('minutes'))
        ]),
        Row([
            Col([Row([range_slider("ages", 15, max_age, 15, max_age, 1)])]),
            Col([Row([range_slider("values", 0, max_value, 0, max_value, 1)])]),
            Col([Row([range_slider("minutes", 0, max_mins, 1250, max_mins, 100)])])
        ]),
    ], style={"width": "100%"}, body=True),
    Card([
        Row([
            Col(html.Label('nationality')),
            Col(html.Label('club')),
        ]),
        Row([
            Col([
                drop_down("nation", ['All'] + nations, "All"),
            ]),
            Col([
                drop_down("club", ['All'] + clubs, "All"),
            ]),
        ]),
    ], style={"width": "100%", "height": "50%"}, body=True),
Card([
        Row([
            Col(html.Label('names')),
        ]),
        Row([
            dcc.Dropdown(
                id='names',
                options=names,
                value=[],
                multi=True
            ),
        ]),
    ], style={"width": "100%", "height": "50%"}, body=True),
    Row([
        html.Div([dcc.Graph(id='main-plot', config={'displayModeBar': False})])
    ])
])


@app.callback(
    Output('main-plot', 'figure'),
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
            State('xcat', 'value'),
            State('ycat', 'value'),
            State('zcat', 'value'),
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
        Input('per_min', 'value'),
        Input('names', 'value'),
    ],
    prevent_initial_call=True
)
def update(xyz, xyz_cats, club, nationality, ages, values, minutes, seasons, comps, positions, colour, dim, per_min, names):
    per_min = [bool(per_min and str(axis) not in not_per_min) for axis in tuple(xyz)]
    query = make_query(xyz, xyz_cats, club, nationality, ages, values, minutes, seasons, comps, positions, names, per_min)
    x_label = [x['label'] for x in xyz[3] if x['value'] == xyz[0]][0]
    y_label = [x['label'] for x in xyz[4] if x['value'] == xyz[1]][0]
    z_label = [x['label'] for x in xyz[5] if x['value'] == xyz[2]][0]
    with engine.connect() as conn:
        graph_params = dict(
            data_frame=read_sql(query, con=conn).round({'x': 3, 'y': 3, 'z': 3}).fillna(0),
            x='x',
            y='y',
            color=colour,
            hover_data=[
                'name',
                'nationality',
                'position',
                'current_value',
                'age',
                'club'
            ],
            labels={
                "x": f"({x_label}) / 90" if per_min[0] else x_label,
                "y": f"({y_label}) / 90" if per_min[1] else y_label,
                "z": f"({z_label}) / 90" if per_min[2] else z_label,
                "name": "Name",
                'nationality': "Nation",
                'position': "Position",
                'current_value': "Value",
                'age': "Age",
                'club': "Team"
            }
        )
    conn.close()
    if dim:
        plot = scatter_3d
        graph_params['z'] = 'z'
    else:
        plot = scatter
    fig = plot(**graph_params)
    fig.update_layout(
        height=700,
        autosize=True,
        margin=dict(t=0, b=0, l=0, r=0),
        template="plotly_white",
    )
    fig.update_traces(
        marker_size=5
    )
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=1),
        aspectmode="auto"
    )
    return fig


def select_clause(sub, table):
    if sub in aggregates.keys():
        v = aggregates[sub]
        return cast(func.sum(table.c[v[0]]) /
                    (1.0 * func.coalesce(func.nullif(func.sum(table.c[v[1]]), 0), 1)) * v[2], v[3])
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


def player_sub_query(club, nationality, values, positions, names):
    player = metadata.tables["player"]
    query = select(
        player.c["id"],
        player.c["name"],
        player.c["club"],
        player.c["nationality"],
        player.c["position"],
        player.c["current_value"],
        # func.regexp_replace(player.c['name'], '^.* ', '').label('shortened'),
        extract('year', func.age(player.c["dob"])).label('age').cast(Integer),
    ).where(
        (player.c["club"] == club if club != 'All' else true()) &
        (player.c["nationality"] == nationality if nationality != 'All' else true()) &
        (player.c["current_value"].between(values[0], values[1])) &
        (player.c["position"].in_(positions)) &
        (player.c['name'].in_(names) if names != [] else true())
    ).subquery()
    return query


def mins_sub_query(seasons, comps):
    table = metadata.tables["playingtime"]
    query = select(
        table.c["id"],
        func.sum(table.c["minutes"]).label('mins')
    ).group_by(
        table.c["id"],
    ).where(
        table.c["season"].in_(seasons),
        table.c["comp"].in_(comps)
    ).subquery()
    return query


def make_query(xyz, xyz_cats, club, nationality, ages, values, minutes, seasons, comps, positions, names, per_min):
    player = player_sub_query(club, nationality, values, positions, names)
    x = data_sub_query(xyz[0], xyz_cats[0], seasons, comps, 'x')
    y = data_sub_query(xyz[1], xyz_cats[1], seasons, comps, 'y')
    z = data_sub_query(xyz[2], xyz_cats[2], seasons, comps, 'z')
    mins = mins_sub_query(seasons, comps)

    query = select(
        player.c["id"],
        player.c["name"],
        player.c["club"],
        player.c["nationality"],
        player.c["position"],
        player.c["current_value"],
        player.c["age"],
        # player.c["shortened"],
        (x.c["x"] / mins.c["mins"] * 90).label('x') if per_min[0] else x.c["x"],
        (y.c["y"] / mins.c["mins"] * 90).label('y') if per_min[1] else y.c["y"],
        (z.c["z"] / mins.c["mins"] * 90).label('z') if per_min[2] else z.c["z"]
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
        isouter=True
    ).where(
        player.c["age"].between(ages[0], ages[1]),
        mins.c["mins"].between(minutes[0], minutes[1])
    )
    return query


@app.callback(
    Output('x', 'options'),
    Input('xcat', 'value'),
    prevent_initial_call=True
)
def update_x_dropdown(value):
    return [{"label": key, "value": value} for key, value in axis_options[value].items()]


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
    return [{"label": key, "value": value} for key, value in axis_options[value].items()]


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
    return [{"label": key, "value": value} for key, value in axis_options[value].items()]


@app.callback(
    Output('z', 'value'),
    Input('z', 'options')
)
def update_z_value(options):
    return options[0]['value']


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
