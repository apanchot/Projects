import dash
import dash_bootstrap_components as dbc

external_scripts = ['https://code.jquery.com/jquery-3.3.1.min.js',
                    'https://cdn.datatables.net/v/dt/dt-1.10.18/datatables.min.js']

external_stylesheets = [dbc.themes.BOOTSTRAP]


app = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
