from widgets import radio_item, drop_down, check_list, range_slider
from dash import Dash
from dash_bootstrap_components import Col, Row, Card, themes
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State, ClientsideFunction
from plotly.express import scatter_3d, scatter
from maps import cat_options, axis_options, seasons, comps, positions, not_per_min, aggregates, combined
from pandas import read_sql, DataFrame
from os import environ
from sqlalchemy import text, create_engine, select, MetaData, func, extract, Integer, true, false, cast, or_, and_
from textalloc import allocate_text
from sklearn.ensemble import IsolationForest
from numpy import vectorize

engine = create_engine(environ["SQLALCHEMY_DATABASE_URI"])

metadata = MetaData()
metadata.reflect(engine, only=cat_options.values())
cat_options["Combined"] = "combined"

app = Dash(external_stylesheets=[themes.BOOTSTRAP])
server = app.server

with engine.connect() as conn:
    max_mins = conn.execute(
        text("select max(sum) from (select id, sum(minutes) from playingtime group by (id)) as x")).scalar()
    max_value, max_age = conn.execute(
        text("select ceiling(max(current_value))::Integer,max(date_part('year',age(dob)))::Integer from player")).all()[0]
    clubs = conn.execute(
        text("select distinct club from player where current_value > 20.0 order by club")).scalars().all()
    nations = conn.execute(
        text("select distinct nationality from player order by nationality")).scalars().all()
    names = conn.execute(
        text("select name from player order by name")).scalars().all()
    conn.close()

app.layout = Col([
    html.Div([html.Div(id='x_pixels')], style={'display': 'none'}),
    dcc.Store(id='dataframe', storage_type='session'),
    dcc.Markdown('''
        # FBREF-3D
    
        A dashboard to visualise the data on fbref.com as 2D or 3D scatter graphs. The data is periodically scraped and 
        added to a PostgreSQL database. Annotations are automatically adjusted to avoid overlapping, which can take a 
        few seconds to process. The number of annotations is limited to 50.
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
                    drop_down("x", axis_options["shooting"], "non_penalty_goals")
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
                    drop_down("zcat", cat_options, "combined"),
                    drop_down("z", axis_options["combined"], None)
                ])
            ])
        ]),
    ], style={"width": "100%", "height": "50%"}, body=True),
    Card([
        Row([
            Col(Row(html.Label('scatter type'), justify='center'), width=3),
            Col(Row(html.Label('marker colours'), justify='center'), width=3),
            Col(Row(html.Label('per minute or totals'), justify='center'), width=3),
            Col(Row(html.Label('annotation'), justify='center'), width=2),
            Col(Row(html.Label('no.'), justify='center'), width=1)
        ]),
        Row([
            Col([
                Row([
                    radio_item(id="dimension", options={"2D": True, "3D": False}, value=True)
                ], justify='center')
            ], width=3),
            Col([
                Row([
                    radio_item(id="colour", options={"Z-Axis": "z", "Position": "position"}, value="z")
                ], justify='center')
            ], width=3),
            Col([
                Row([
                    radio_item(id="per_min", options={"Per 90": True, "Total": False}, value=True)
                ], justify='center')
            ], width=3),
            Col([
                Row([
                    radio_item(id="annotation", options=["outliers", "none"], value="outliers")
                ], justify='center')
            ], width=2),
            Col([
                Row([
                    dcc.Input(id='outliers', type='number', value=25, size='2', max=50, min=0, step=1)
                ], justify='center')
            ], width=1)
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
            Col(dcc.Dropdown(id='names', options=names, value=[], multi=True), width=10),
            Col(radio_item(id="add-only", options={"add": True, "only": False}, value=True), width=2),
        ]),
    ], style={"width": "100%", "height": "50%"}, body=True),
    Row([
        html.Div([dcc.Graph(id='main-plot', config={'displayModeBar': False})])
    ])
])

app.clientside_callback(
    """
    function(a,b,c,d,e,f,g,h,i) {
        var w = window.innerWidth;
        return w;
    }
    """,
    Output('x_pixels', 'children'),
    Input('dataframe', 'data'),
    Input('colour', 'value'),
    Input('dimension', 'value'),
    Input('main-plot', 'selectedData'),
    Input('add-only', 'value'),
    Input('outliers', 'value'),
    Input('annotation', 'value')
)


@app.callback(
    Output('dataframe', 'data'),
    inputs=[
        (
            Input('x', 'value'),
            Input('y', 'value'),
            Input('z', 'value'),
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
        Input('names', 'value'),
        Input('per_min', 'value'),
        Input('add-only', 'value')
    ]
)
def get_dataframe(xyz, xyz_cats, club, nation, ages, values, minutes, seasons, comps, positions, names, per_min, add):
    per_min = [bool(per_min and str(axis) not in not_per_min) for axis in tuple(xyz)]
    query = make_query(xyz, xyz_cats, club, nation, ages, values, minutes, seasons, comps, positions, names, per_min, add)
    with engine.connect() as conn:
        df = read_sql(query, con=conn).round({'x': 3, 'y': 3, 'z': 3}).fillna(0)
        conn.close()
    return df.to_dict('records')


@app.callback(
    Output('main-plot', 'figure'),
    inputs=[
        (
            State('x', 'value'),
            State('y', 'value'),
            State('z', 'value'),
            State('x', 'options'),
            State('y', 'options'),
            State('z', 'options'),
        ),
        State('dataframe', 'data'),
        State('dimension', 'value'),
        State('per_min', 'value'),
        State('outliers', 'value'),
        State('annotation', 'value'),
        State('colour', 'value'),
        Input('x_pixels', 'children'),
    ],
    prevent_initial_call=True
)
def update(xyz, data, dim, per_min, outliers, annotation, colour, x_pixels):
    per_min = [bool(per_min and str(axis) not in not_per_min) for axis in tuple(xyz)]
    x_label = [x['label'] for x in xyz[3] if x['value'] == xyz[0]][0]
    y_label = [x['label'] for x in xyz[4] if x['value'] == xyz[1]][0]
    z_label = [x['label'] for x in xyz[5] if x['value'] == xyz[2]][0]
    df = DataFrame(data)
    graph_params = dict(
        data_frame=df,
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
        },
    )
    if dim:
        plot = scatter
    else:
        plot = scatter_3d
        graph_params['z'] = 'z'
    fig = plot(**graph_params)
    if annotation == 'outliers':
        l = len(df.index)
        if l > 50:
            isf = IsolationForest(n_estimators=100, random_state=42, contamination=0.5 if outliers/l > 0.5 else outliers/l)
            preds = isf.fit_predict(df[['x', 'y']])
            df["iso_forest_outliers"] = preds
            df = df[df["iso_forest_outliers"] == -1]
            df['name'] = vectorize(lambda x: x.split(' ')[-1])(df['name'])
        # the right margin width is linearly interpolated and subtracted from the screen width (x_pixels)
        allocate_text(
            df['x'].tolist(),
            df['y'].tolist(),
            df['name'].tolist(),
            fig,
            x_pixels - (53 + 93 + (121 - 93) * (x_pixels - 450) / (1920 - 450)),
            700,
            graph_params['data_frame']['x'].tolist(),
            graph_params['data_frame']['y'].tolist(),
        )
    fig.update_layout(
        height=700,
        autosize=True,
        margin=dict(t=0, b=0, l=0, r=0),
        template="plotly_white",
        font_family='Arial'
    )
    fig.update_traces(
        marker_size=5,
    )
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=1),
        aspectmode="auto"
    )
    fig.update_coloraxes(
        colorbar_title=dict(
            side='right'
        )
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
    if cat == "combined":
        return multi_table_sub_query(sub, seasons, comps, label)
    else:
        return single_table_sub_query(sub, cat, seasons, comps, label)


def single_table_sub_query(sub, cat, seasons, comps, label):
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


def multi_table_sub_query(sub, seasons, comps, label):
    v = combined[sub]
    table1 = metadata.tables[str(v[2])]
    table2 = metadata.tables[str(v[3])]
    a = f'{label}a'
    b = f'{label}b'

    query1 = select(
        table1.c["id"],
        func.sum(table1.c[v[0]]).label(a)
    ).group_by(
        table1.c["id"]
    ).where(
        table1.c["season"].in_(seasons),
        table1.c["comp"].in_(comps)
    ).subquery()

    query2 = select(
        table2.c["id"],
        func.sum(table2.c[v[1]]).label(b)
    ).group_by(
        table2.c["id"]
    ).where(
        table2.c["season"].in_(seasons),
        table2.c["comp"].in_(comps)
    ).subquery()

    combined_query = select(
        query1.c["id"],
        (query1.c[a] + query2.c[b]).label(label)
    ).join(
        query2,
        query1.c["id"] == query2.c["id"]
    ).subquery()
    return combined_query


def player_sub_query(club, nationality, values, positions, names, add):
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
        or_(
            and_(
                (player.c["club"] == club if club != 'All' else true()),
                (player.c["nationality"] == nationality if nationality != 'All' else true()),
                (player.c["current_value"].between(values[0], values[1])),
                (player.c["position"].in_(positions)),
            ),
            (player.c['name'].in_(names) if names != [] else false()),
        )
        if add else player.c['name'].in_(names),
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


def make_query(xyz, xyz_cats, club, nationality, ages, values, minutes, seasons, comps, positions, names, per_min, add):
    player = player_sub_query(club, nationality, values, positions, names, add)
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
    app.run_server(debug=False, use_reloader=False)
