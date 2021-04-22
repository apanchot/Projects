import dash_bootstrap_components as dbc


navbar_style = {
    'left': 'auto',
    'width': '100%',
    'padding': '2rem 1rem',
    'font-size': '150%',
}

def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/home")),
            dbc.NavItem(dbc.NavLink("CPV", href="/codes")),
            dbc.NavItem(dbc.NavLink("Country", href="/countries")),
            dbc.NavItem(dbc.NavLink("Business", href="/businesses")),
        ],
        brand="EU Procurements Explorer",
        color="primary",
        dark=True,
        style=navbar_style
    )
    return navbar