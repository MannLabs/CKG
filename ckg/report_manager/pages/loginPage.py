import dash
from dash import html, dcc

title = "CKG login"
subtitle = ""
description = ""

dash.register_page(__name__, path="/apps/loginPage", title=f"{title} - {subtitle}", description=description)


def layout():
    login_layout = html.Div(children=[
        html.H1(children=title),
        html.H2(children=subtitle),
        html.Div(children=description),

        html.Div([html.Form([
            dcc.Input(placeholder='username', name='username', type='text', id="username-box"),
            dcc.Input(placeholder='password', name='password', type='password', id="password-box"),
            html.Button('Login', type='submit')], action='/apps/login', method='post', id="login-button")])
    ])
    return login_layout
