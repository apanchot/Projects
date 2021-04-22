import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from app import app
import base64
import backend.performance_evaluation as perf_eval

layout = html.Div([
    html.Div([
        html.H1('Homepage'),
        html.H3(id='textarea-count', style={'whiteSpace': 'pre-line'}),
    ],style={'text-align': 'center', 'margin-top': '2%'}),

    html.Div([
        html.Div([
            dcc.Link('CPV', href='/codes', style={'font-size': '150%'}),
        ], style={'flex': '33%', 'text-align': 'center'}),
        html.Div([
            dcc.Link('Country', href='/countries', style={'font-size': '150%'}),
        ], style={'flex': '33%', 'text-align': 'center'}),
        html.Div([
            dcc.Link('Business', href='/businesses', style={'font-size': '150%'}),
        ], style={'flex': '33%', 'text-align': 'center'}),
    ], style={'display': 'flex', 'margin': 'auto', 'width': '980px', 'height': '85px', 'margin-top': '2%'}),
    html.Hr(), html.Hr(), html.Hr(),  
    html.Div([
        html.H1('Performance test'),
        html.Div('This will run all the dashboard queries sequentially and output the time taken by all queries'),
        html.Button('Start evaluation', id='buttonEval', style = {'margin-top': '2%'}),
        dcc.Interval(id="progress-interval", n_intervals=0, interval=2000, max_intervals=1000),
        dbc.Progress(id="progress", striped=True, animated=True, style={"height": "40px", 'margin-top': '2%'}),
    ],style={'text-align': 'center', 'margin-top': '2%'}),
    
    html.Hr(), html.Hr(), html.Hr(),  
    html.H1('File Upload', style={'text-align': 'center'}),
    html.Div('Allows for new document uploads and measures the time the upload took to complete', style={'text-align': 'center'}),
    html.Div(id='textarea-avg', style={'whiteSpace': 'pre-line'}),
    html.Div(id='textarea-time', style={'whiteSpace': 'pre-line'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    json_obj = base64.b64decode(content_string)

    try:
        if 'json' not in filename:
            raise Exception("Invalid data file")
        
        (inserted_ids, processed_time) = perf_eval.insert_json(json_obj)
    except Exception as e:
        print(f'There was an error processing this file: {e}')
        return html.Div([
            f'There was an error processing this file: {e}'
        ])

    return html.Div([
        html.H6(f"Inserted ids {inserted_ids}"),
        html.Hr(),  # horizontal line
        html.H3(f"Insertion time: {processed_time:.3f} seconds!"),

    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = parse_contents(list_of_contents, list_of_names, list_of_dates) 
        return children

@app.callback(Output('textarea-count', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output_textarea_count(content, name, date):
    return f"{perf_eval.get_collection_count()} contracts on the database"

@app.callback(Output('textarea-avg', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_textarea_avg(list_of_contents, list_of_names, list_of_dates):
    return f"Collection stats: {perf_eval.get_collection_stats()}"

@app.callback(
        Output("buttonEval", "disabled")
    ,[
        Input("buttonEval", "n_clicks")
    ])
def start_evaluation(n_clicks):
    if n_clicks is None:
        return False

    from threading import Thread
    Thread(target=perf_eval.performance_evaluation).start()
    return True

@app.callback(
    [
        Output("progress", "value"), 
        Output("progress", "children"), 
    ],[
        Input("progress-interval", "n_intervals")
    ],
)
def update_progress(n):
    disabled = True if n == 50 else False
    try:
        with open(".query.state", 'r') as file:
            data = file.read().split(":")
    except:
        data = (0, "Not started")
    return data[0], data[1]