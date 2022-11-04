import dash
from dash import html

title = "Welcome to the app"
subtitle = "You are successfully authorized"
description = ""

dash.register_page(__name__, path='/apps/initial', title=f"{title} - {subtitle}", description=description)


def layout():
    inital_layout = [html.H1(children=title),
                     html.H2(children=subtitle),
                     html.Div(children=description)]
    return inital_layout
