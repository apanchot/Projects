import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from app import app
import apps.dcc_functions as f

########################################################################################################################


layout = html.Div([
    html.Div([
        html.Br(),
        html.Div([
            html.H4('Average Company Expenditure'),
            dbc.Spinner(html.H4(id='business_box_1', className='box_1'))
        ], style={'margin': 'auto', 'width': '25%', 'height': '25%', 'margin-top': '20px'}),
        html.Div([
            html.Div([
                html.H4('Average Company Contract Count'),
                dbc.Spinner(html.H4(id='business_box_2', className='box_2'))
            ], style={'width': '25%', 'margin-left': '10%'}),
            html.Div([
                html.H4('Average Company Contract Offers'),
                dbc.Spinner(html.H4(id='business_box_3', className='box_3'))
            ], style={'width': '25%', 'margin-left': '30%'}),
        ], style={'display': 'flex', 'height': '25%', 'margin-top': '20px'}),
        html.Div([
            html.Div([
                html.H4('Average Company Expenditure with EU Funds'),
                dbc.Spinner(html.H4(id='business_box_4', className='box_4'))
            ], style={'width': '25%', 'margin-left': '24%'}),
            html.Div([
                html.H4('Average Company Expenditure without EU Funds'),
                dbc.Spinner(html.H4(id='business_box_5', className='box_5'))
            ], style={'width': '25%', 'margin-left': '2%'}),
        ], style={'display': 'flex', 'height': '25%', 'margin-top': '20px'}),
    ], style={'margin': 'auto', 'height': '520px', 'margin-top': '50px'}),

    html.Br(),
    html.Br(),
    html.Br(),

    html.H3('Most active Companies', style={'height': '10%', 'text-align': 'center'}),

    html.Br(),

    html.Div([
        dbc.Spinner(dcc.Graph(id='business_treemap'))
    ], style={'height': '10%', 'text-align': 'center'}),

    html.Br(),
    html.Br(),
    html.Br(),

    html.H3('Companies Average Expenditure', style={'height': '10%', 'text-align': 'center'}),

    html.Br(),

    html.Div([
        html.Div([
            html.Div([html.H4('The Highest Spending Companies'), dbc.Spinner(dcc.Graph(id='business_bar_1'))], style={'width': '45%'}),
            html.Div([], style={'width': '10%'}),
            html.Div([html.H4('The Lowest Spending Companies'), dbc.Spinner(dcc.Graph(id='business_bar_2'))], style={'width': '45%'}),
        ], style={'display': 'flex'})
    ]),

    html.Br(),
    html.Br(),
    html.Br(),

    html.H3('Where are the highest Spending Companies of each Country?', style={'height': '10%', 'text-align': 'center'}),

    html.Br(),

    html.Div([
        dbc.Spinner(dcc.Graph(id='business_map'))
    ], style={'height': '10%', 'text-align': 'center'}),

    html.Br(),
    html.Br(),
    html.Br(),

    html.H3('Top Co-Occurring Companies in Contracts', style={'height': '10%', 'text-align': 'center'}),

    html.Br(),

    html.Div([
        dbc.Spinner(dcc.Graph(id='business_connection'))
    ], style={'height': '10%', 'text-align': 'center'})

])


@app.callback(
    Output("button_business", "n_clicks"),
    [
        Input('url', 'pathname')
    ]
)
def callbacks(none):
    return


@app.callback(
    [
        Output("business_box_1", "children"),
        Output("business_box_2", "children"),
        Output("business_box_3", "children"),
        Output("business_box_4", "children"),
        Output("business_box_5", "children")
    ],
    [
        Input('button_business', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    boxes = f.business_box(bot_year, top_year, country_list)
    print('Queried business Boxes')
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
    Output("business_bar_1", "figure"),
    [
        Input('button_business', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.business_bar_1(bot_year, top_year, country_list)
    print('Queried business Bar 1!')
    return fig

@app.callback(
    Output("business_bar_2", "figure"),
    [
        Input('button_business', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.business_bar_2(bot_year, top_year, country_list)
    print('Queried business bar 2')
    return fig

@app.callback(
    Output("business_treemap", "figure"),
    [
        Input('button_business', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.business_treemap(bot_year, top_year, country_list)
    print('Queried business treemap')
    return fig


@app.callback(
    Output("business_map", "figure"),
    [
        Input('button_business', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.business_map(bot_year, top_year, country_list)
    print('Queried business Map')
    return fig


@app.callback(
    Output("business_connection", "figure"),
    [
        Input('button_business', 'n_clicks')
    ],
    [
        State('year_slider', 'value'),
        State('country_drop', 'value')
    ]
)
def callbacks(n_clicks_1, year, country_list):
    bot_year = year[0]
    top_year = year[1]

    fig = f.business_connection(bot_year, top_year, country_list)
    print('Queried business Connection')
    return fig