import dash
import flask
from dash import html, dcc

title = "CKG Admin Dashboard"
subtitle = ""
description = ""

dash.register_page(__name__, path='/apps/admin', title=f"{title} - {subtitle}", description=description)


def layout():
    # TODO: adapt authentication for admin
    session_cookie = flask.request.cookies.get('custom-auth-session')
    logged_in = session_cookie is not None
    if logged_in == False:
        return html.Div(["Please ", dcc.Link("login", href="/apps/loginPage"), " to continue"])

    create_user_form = [html.H3("Create CKG User"), html.Form([
        html.Div(children=[html.Label('Name'),
                           dcc.Input(placeholder='name', name='name', type='text', required=True),
                           html.Label('Surname'),
                           dcc.Input(placeholder='surname', name='surname', type='text', required=True),
                           html.Label('Acronym'),
                           dcc.Input(placeholder='acronym', name='acronym', type='text'),
                           html.Label('Affiliation'),
                           dcc.Input(placeholder='affiliation', name='affiliation', type='text', required=True)]),
        html.Div(children=[html.Label('E-mail'),
                           dcc.Input(placeholder='email', name='email', type='email', required=True),
                           html.Label('alternative E-mail'),
                           dcc.Input(placeholder='alt email', name='alt_e-mail', type='email'),
                           html.Label('Phone number'),
                           dcc.Input(placeholder='phone', name='phone', type='tel', required=True)]),
        html.Div(children=[html.Button('CreateUser', type='submit', className='button_link')],
                 style={'width': '100%', 'padding-left': '87%', 'padding-right': '0%'})], action='/create_user',
        method='post')]
    update_database_bts = [html.H3("Build CKG Database"),
                           html.Div(children=[
                               html.Form([html.Button('Minimal Update', type='submit', className='button_link')],
                                         action='/update_minimal', method='post'),
                               html.P(
                                   "This option will load into CKG's graph database the licensed Ontologies and Databases and all their missing relationships.",
                                   className='description_p')]),
                           html.Br(),
                           html.Div(children=[
                               html.Form([html.Button('Full Update', type='submit', className='button_link'),
                                          html.Div(children=[html.H4("Download:"),
                                                             html.Label("Yes", className='radioitem'),
                                                             dcc.Input(id='Yes',
                                                                       name='dwn-radio',
                                                                       value=True,
                                                                       type='radio'),
                                                             html.Label("No", className='radioitem'),
                                                             dcc.Input(id='No',
                                                                       name='dwn-radio',
                                                                       value=False,
                                                                       type='radio')])], action='/update_full',
                                         method='post'),
                               html.P(
                                   "This option will regenerate the entire database, downloading data from the different Ontologies and Databases (Download=Yes) and loading them and all existing projects into CKG's graph database.",
                                   className='description_p')])]
    admin_options = html.Div(children=[html.Div(children=create_user_form, className='div_framed'),
                                       html.Div(children=update_database_bts, className='div_framed')])

    admin_layout = [html.Div(children=[
        html.H1(children=title),
        html.H2(children=subtitle),
        html.Div(children=description),
    ])]

    admin_layout.extend(admin_options)
    return admin_layout
