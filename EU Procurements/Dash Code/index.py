import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from apps import home, codes, countries, businesses
from apps.sidebar import render_sidebar
from apps.navbar import Navbar
import pandas as pd
from app import app

server = app.server

sidebar = render_sidebar()
navbar = Navbar()

content = html.Div(id="content", className='content')

app.layout = html.Div([dcc.Location(id="url", refresh=False), navbar, sidebar, content])


@app.callback(Output("content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/home"]:
        return home.layout
    elif pathname == "/codes":
        return codes.layout
    elif pathname == "/countries":
        return countries.layout
    elif pathname == "/businesses":
        return businesses.layout
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')