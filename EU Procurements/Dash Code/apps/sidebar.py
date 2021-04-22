import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


countries =[
    ['MK', 'North Macedonia'],
    ['LT', 'Lithuania'],
    ['SI', 'Slovenia'],
    ['CY', 'Cyprus'],
    ['NL', 'Netherlands'],
    ['LI', 'Liechtenstein'],
    ['HR', 'Croatia'],
    ['AT', 'Austria'],
    ['RO', 'Romania'],
    ['ES', 'Spain'],
    ['SE', 'Sweden'],
    ['DK', 'Denmark'],
    ['FI', 'Finland'],
    ['PL', 'Poland'],
    ['DE', 'Germany'],
    ['GR', 'Greece'],
    ['FR', 'France'],
    ['CH', 'Switzerland'],
    ['BG', 'Bulgaria'],
    ['HU', 'Hungary'],
    ['BE', 'Belgium'],
    ['SK', 'Slovakia'],
    ['NO', 'Norway'],
    ['IT', 'Italy'],
    ['IS', 'Iceland'],
    ['EE', 'Estonia'],
    ['LV', 'Latvia'],
    ['CZ', 'Czechia'],
    ['MT', 'Malta'],
    ['LU', 'Luxembourg'],
    ['UK', 'United Kingdom'],
    ['IE', 'Ireland'],
    ['PT', 'Portugal']
]

country_options = [dict(label=country[1], value=country[0]) for country in countries]


image_filename = 'assets/eu_icon.png'


def render_sidebar():
    sidebar = html.Div(
        [
            html.Hr(),
            html.H2('Year Slider', style={'color': 'white'}),
            dcc.RangeSlider(
                id='year_slider',
                min=2008,
                max=2020,
                value=[2008, 2020],
                marks={2008: '2008',
                       2010: '2010',
                       2012: '2012',
                       2014: '2014',
                       2016: '2016',
                       2018: '2018',
                       2020: '2020'},
                step=1,
            ),
            html.Br(),
            html.Button('Submit', id='button_code', className='button_code', style={'margin-left': '60%'}),
            html.Button('Submit', id='button_country', className='button_country', style={'margin-left': '60%'}),
            html.Button('Submit', id='button_business', className='button_business', style={'margin-left': '60%'}),
            html.Br(),
            html.H2('Country Choice', style={'color': 'white'}),
            dcc.Dropdown(
                id='country_drop',
                options=country_options,
                value=[country[0] for country in countries],
                multi=True,
                style={'max-height': '400px', 'overflow-y': 'scroll', 'background-color':'#003399', 'color':'#f9f9f9'}
            )
        ], className='sidebar', id='sidebar',
    )

    return sidebar
