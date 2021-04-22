import dash_core_components as dcc
import dash_table as dt
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from app import app
import apps.dcc_functions as f


cpv_division = [
    ['Clothing, footwear, luggage articles and accessories', '18'],
    ['Leather and textile fabrics, plastic and rubber materials',
    '19'],
    ['Installation services (except software)', '51'],
    ['Construction structures and materials; auxiliary products to construction (except electric apparatus)',
    '44'],
    ['Office and computing machinery, equipment and supplies except furniture and software packages',
    '30'],
    ['Research and development services and related consultancy services',
    '73'],
    ['Repair and maintenance services', '50'],
    ['Medical equipments, pharmaceuticals and personal care products',
    '33'],
    ['Transport equipment and auxiliary products to transportation',
    '34'],
    ['Health and social work services', '85'],
    ['Financial and insurance services', '66'],
    ['Electrical machinery, apparatus, equipment and consumables; lighting',
    '31'],
    ['Education and training services', '80'],
    ['Real estate services', '70'],
    ['Public utilities', '65'],
    ['IT services: consulting, software development, Internet and support',
    '72'],
    ['Sewage, refuse, cleaning and environmental services', '90'],
    ['Food, beverages, tobacco and related products', '15'],
    ['Business services: law, marketing, consulting, recruitment, printing and security',
    '79'],
    ['Security, fire-fighting, police and defence equipment', '35'],
    ['Services related to the oil and gas industry', '76'],
    ['Petroleum products, fuel, electricity and other sources of energy',
    '09'],
    ['Administration, defence and social security services', '75'],
    ['Radio, television, communication, telecommunication and related equipment',
    '32'],
    ['Agricultural, farming, fishing, forestry and related products',
    '03'],
    ['Recreational, cultural and sporting services', '92'],
    ['Other community, social and personal services', '98'],
    ['Construction work', '45'],
    ['Supporting and auxiliary transport services; travel agencies services',
    '63'],
    ['Laboratory, optical and precision equipments (excl. glasses)',
    '38'],
    ['Agricultural, forestry, horticultural, aquacultural and apicultural services',
    '77'],
    ['Furniture (incl. office furniture), furnishings, domestic appliances (excl. lighting) and cleaning products',
    '39'],
    ['Software package and information systems', '48'],
    ['Transport services (excl. Waste transport)', '60'],
    ['Agricultural machinery', '16'],
    ['Architectural, construction, engineering and inspection services',
    '71'],
    ['Hotel, restaurant and retail trade services', '55'],
    ['Musical instruments, sport goods, games, toys, handicraft, art materials and accessories',
    '37'],
    ['Postal and telecommunications services', '64'],
    ['Collected and purified water', '41'],
    ['Machinery for mining, quarrying, construction equipment', '43'],
    ['Industrial machinery', '42'],
    ['Mining, basic metals and related products', '14'],
    ['Printed matter and related products', '22'],
    ['Chemical products', '24']
]

cpv_options = [dict(label=cpv[0], value=cpv[1]) for cpv in cpv_division]

cpv_dropdown = dcc.Dropdown(
                id='cpv_drop',
                options=cpv_options,
                value=cpv_division[0][1],
                multi=False
            ),


########################################################################################################################


layout = html.Div([

    html.Div([

        html.Br(),
        html.Div([
            html.H4('Average CPV Division Expenditure'),
            dbc.Spinner(html.H4(id='box_1', className='box_1'))
        ], style={'margin': 'auto', 'width': '25%', 'height': '25%', 'margin-top': '20px'}),
        html.Div([
            html.Div([
                html.H4('Average CPV Division Contract Count'),
                dbc.Spinner(html.H4(id='box_2', className='box_2'))
            ], style={'width': '25%', 'margin-left': '10%'}),
            html.Div([
                html.H4('Average CPV Division Contract Offers'),
                dbc.Spinner(html.H4(id='box_3', className='box_3'))
            ], style={'width': '25%', 'margin-left': '30%'}),
        ], style={'display': 'flex', 'height': '25%', 'margin-top': '20px'}),
        html.Div([
            html.Div([
                html.H4('Average CPV Division Expenditure with EU Funds'),
                dbc.Spinner(html.H4(id='box_4', className='box_4'))
            ], style={'width': '25%', 'margin-left': '24%'}),
            html.Div([
                html.H4('Average CPV Division Expenditure without EU Funds'),
                dbc.Spinner(html.H4(id='box_5', className='box_5'))
            ], style={'width': '25%', 'margin-left': '2%'}),
        ], style={'display': 'flex', 'height': '25%', 'margin-top': '20px'}),
    ], style={'margin': 'auto', 'height': '520px', 'margin-top': '50px'}),

    html.H3('CPV Division Contract Counts', style={'height': '10%', 'text-align': 'center', 'margin-top': '100px'}),
    html.Br(),
    dbc.Spinner(dcc.Graph(id='treemap')),

    html.Br(),
    html.Br(),
    html.Br(),

    html.H3('CPV Division Average Expenditure', style={'height': '10%', 'text-align': 'center'}),

    html.Div([
        html.Div([
            html.Div([html.H4('The Highest value Divisions'), dbc.Spinner(dcc.Graph(id='bar_1'))], style={'width': '45%'}),
            html.Div([], style={'width': '10%'}),
            html.Div([html.H4('The Lowest value Divisions'), dbc.Spinner(dcc.Graph(id='bar_2'))], style={'width': '45%'}),
        ], style={'display': 'flex'})
    ]),

    html.Br(),
    html.Br(),
    html.Br(),

    html.H3('Countries Most Profitable CPV Division', style={'height': '10%', 'text-align': 'center'}),

    html.Div([
        dbc.Spinner(dcc.Graph(id='cpv_map'))
    ], style={'height': '10%', 'text-align': 'center'}),

    html.Br(),
    html.Br(),
    html.Br(),

    html.H3('Contract Distribution by CPV Division', style={'height': '10%', 'text-align': 'center'}),
    html.Div([
        html.Div([
            html.Div(cpv_dropdown, style={'height': '20%'}),
            html.Br(),
            html.Div([dbc.Spinner(dcc.Graph(id='hist'))], style={'height': '80%'}),
        ])
    ]),

    html.Br(),
    html.Br(),
    html.Br(),

    html.H3('European Investment on CPV Divisions ', style={'height': '10%', 'text-align': 'center'}),

    html.Br(),

    html.Div([
        html.Div([
            html.Div([html.H4('Highest Average CPV Divisions with EU Funds'), dbc.Spinner(dcc.Graph(id='bar_3'))], style={'width': '45%'}),
            html.Div([], style={'width': '5%'}),
            html.Div([html.H4('Highest Average CPV Divisions without EU Funds'), dbc.Spinner(dcc.Graph(id='bar_4'))], style={'width': '50%'}),
        ], style={'display': 'flex'})
    ]),

    html.Br(),
    html.H3('Discrepancies Between Contract award and execution', style={'height': '10%', 'text-align': 'center'}),
    html.Br(),
    html.Div([
        html.H4('Time and Money'),
        dbc.Spinner(dcc.Graph(id='cpv_bar_diff'))
    ], style={}),

])


@app.callback(
    Output("button_code", "n_clicks"),
    [
        Input('url', 'pathname')
    ]
)
def callbacks(none):
    return


@app.callback(
    [
        Output("box_1", "children"),
        Output("box_2", "children"),
        Output("box_3", "children"),
        Output("box_4", "children"),
        Output("box_5", "children")
    ],
    [
        Input('button_code', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    boxes = f.cpv_box(bot_year, top_year, country_list)
    print('Queried cpv Boxes')
    box_1 = boxes[0]
    box_2 = boxes[1]
    box_3 = boxes[2]
    box_4 = boxes[3]
    box_5 = boxes[4]

    return str(box_1) + '€', \
           str(box_2), \
           str(box_3), \
           str(box_4) + '€', \
           str(box_5) + '€'


@app.callback(
    Output("treemap", "figure"),
    [
        Input('button_code', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.cpv_treemap(bot_year, top_year, country_list)
    print('Queried cpv Treemap!')
    return fig

@app.callback(
    Output("bar_1", "figure"),
    [
        Input('button_code', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.cpv_bar_1(bot_year, top_year, country_list)
    print('Queried cpv bar 1')
    return fig

@app.callback(
    Output("bar_2", "figure"),
    [
        Input('button_code', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.cpv_bar_2(bot_year, top_year, country_list)
    print('Queried cpv bar 2')
    return fig


@app.callback(
    Output("bar_3", "figure"),
    [
        Input('button_code', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.cpv_bar_3(bot_year, top_year, country_list)
    print('Queried cpv bar 3')
    return fig


@app.callback(
    Output("bar_4", "figure"),
    [
        Input('button_code', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.cpv_bar_4(bot_year, top_year, country_list)
    print('Queried cpv bar 4')
    return fig


@app.callback(
    Output("hist", "figure"),
    [
        Input('button_code', 'n_clicks')
    ],
    [
        State('cpv_drop', 'value'),
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, cpv, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.cpv_histogram(bot_year, top_year, country_list, cpv)
    print('Queried cpv histogram')
    return fig



@app.callback(
    Output("cpv_map", "figure"),
    [
        Input('button_code', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.cpv_map(bot_year, top_year, country_list)
    print('Queried cpv Map')
    return fig


@app.callback(
    Output("cpv_bar_diff", "figure"),
    [
        Input('button_code', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.cpv_bar_diff(bot_year, top_year, country_list)
    print('Queried cpv Bar Diff')
    return fig
