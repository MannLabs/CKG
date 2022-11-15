import dash
from dash import html, dcc

dash.register_page(__name__, path="/apps/logoutPage")


def layout():
    return html.Div(
        [
            html.Div(html.H2("You have been logged out.")),
            html.Br(),
            dcc.Link("Login", href="/apps/loginPage"),
        ]
    )
