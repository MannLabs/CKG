import base64
import time

import dash
import flask
import numpy as np
import pandas as pd
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

from ckg.graphdb_builder import builder_utils
from ckg.graphdb_connector import connector
from ckg.report_manager.apps import projectCreation
from ckg.report_manager.worker import create_new_project

title = "Project Creation"
subtitle = ""
description = ""

dash.register_page(__name__, path='/apps/projectCreationApp', title=f"{title} - {subtitle}", description=description)

DataTypes = ['clinical', 'proteomics',
             'interactomics', 'phosphoproteomics',
             'longitudinal_proteomics', 'longitudinal_clinical']


def layout():
    session_cookie = flask.request.cookies.get('custom-auth-session')
    logged_in = session_cookie is not None
    if logged_in == False:
        return html.Div(["Please ", dcc.Link("login", href="/apps/loginPage"), " to continue"])

    data = get_data()

    if data == None:
        database_offline_error_layout = [html.Div(children=html.H1("Database is offline", className='error_msg'))]
        return database_offline_error_layout
    else:
        users, tissues, diseases = data
        project_creation_layout = [html.Div([
            html.H1(children=title),
            html.H2(children=subtitle),
            html.Div(children=description),
            html.Div([html.H4('Project information',
                              style={'width': '15.5%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                      html.H4('', id='update_project_id',
                              style={'width': '15%', 'verticalAlign': 'top', 'display': 'none'}),
                      html.Br(),
                      html.Div(children=[html.Label('Project name:*', style={'marginTop': 15}),
                                         dcc.Input(id='project name', placeholder='Insert name...', type='text',
                                                   style={'width': '100%', 'height': '35px'})],
                               style={'width': '100%'}),
                      html.Br(),
                      html.Div(children=[html.Label('Project Acronym:', style={'marginTop': 15}),
                                         dcc.Input(id='project acronym', placeholder='Insert name...', type='text',
                                                   style={'width': '100%', 'height': '35px'})],
                               style={'width': '100%'}),
                      html.Br(),
                      html.Div(children=[html.Label('Project Responsible:*', style={'marginTop': 15})],
                               style={'width': '49%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                      html.Div(children=[html.Label('Project Participants:*', style={'marginTop': 15})],
                               style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top',
                                      'display': 'inline-block'}),
                      html.Div(children=[
                          dcc.Dropdown(id='responsible-picker', options=[{'label': i, 'value': i} for i in users],
                                       value=[], multi=True, searchable=True, style={'width': '100%'})],
                          style={'width': '49%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                      html.Div(children=[
                          dcc.Dropdown(id='participant-picker', options=[{'label': i, 'value': i} for i in users],
                                       value=[], multi=True, searchable=True, style={'width': '100%'})],
                          style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top',
                                 'display': 'inline-block'}),
                      html.Br(),
                      html.Br(),
                      html.Div(children=[html.Label('Project Data Types:*', style={'marginTop': 10})],
                               style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top',
                                      'display': 'inline-block'}),
                      html.Div(children=[html.Label('Project Disease:*', style={'marginTop': 10})],
                               style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top',
                                      'display': 'inline-block'}),
                      html.Div(children=[
                          dcc.Dropdown(id='data-types-picker', options=[{'label': i, 'value': i} for i in DataTypes],
                                       value=[], multi=True, searchable=True, style={'width': '100%'})],
                          style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top',
                                 'display': 'inline-block'}),
                      html.Div(children=[
                          dcc.Dropdown(id='disease-picker', options=[{'label': i, 'value': i} for i in diseases],
                                       value=[], multi=True, searchable=True, style={'width': '100%'})],
                          style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top',
                                 'display': 'inline-block'}),
                      html.Br(),
                      html.Br(),
                      html.Div(children=[html.Label('Project Tissue:*', style={'marginTop': 10})],
                               style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top',
                                      'display': 'inline-block'}),
                      html.Div(children=[html.Label('Project Intervention:', style={'marginTop': 10})],
                               style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top',
                                      'display': 'inline-block'}),
                      html.Div(children=[
                          dcc.Dropdown(id='tissue-picker', options=[{'label': i, 'value': i} for i in tissues],
                                       value=[], multi=True, searchable=True, style={'width': '100%'})],
                          style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top',
                                 'display': 'inline-block'}),
                      html.Div(children=[dcc.Input(id='intervention-picker',
                                                   placeholder='E.g. SNOMED identifier|SNOMED identifier|...',
                                                   type='text', style={'width': '100%', 'height': '54px'})],
                               style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top',
                                      'display': 'inline-block'}),
                      html.Br(),
                      html.Br(),
                      html.Div(children=[html.Label('Timepoints:', style={'marginTop': 15}),
                                         dcc.Input(id='number_timepoints',
                                                   placeholder='E.g. 2 months|15 days|24 hours...', type='text',
                                                   style={'width': '100%', 'height': '35px'})],
                               style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top',
                                      'display': 'inline-block'}),
                      html.Br(),
                      html.Br(),
                      html.Div(children=[html.Label('Follows up project:', style={'marginTop': 15}),
                                         dcc.Input(id='related_to', placeholder='Use the Project Identifier (P000000X)',
                                                   type='text', style={'width': '100%', 'height': '35px'})],
                               style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top',
                                      'display': 'inline-block'}),
                      html.Br(),
                      html.Br(),
                      html.Div(children=[html.Label('Project Description:', style={'marginTop': 15}),
                                         dcc.Textarea(id='project description', placeholder='Enter description...',
                                                      style={'width': '100%', 'height': '100px'})]),
                      html.Br(),
                      html.Div(children=[html.Label('Starting Date:', style={'marginTop': 10}),
                                         dcc.DatePickerSingle(id='date-picker-start', placeholder='Select date...',
                                                              clearable=True)],
                               style={'width': '30%', 'verticalAlign': 'top', 'marginTop': 10,
                                      'display': 'inline-block'}),
                      html.Div(children=[html.Label('Ending Date:', style={'marginTop': 10}),
                                         dcc.DatePickerSingle(id='date-picker-end', placeholder='Select date...',
                                                              clearable=True)],
                               style={'width': '30%', 'verticalAlign': 'top', 'marginTop': 10,
                                      'display': 'inline-block'}),
                      html.Div(children=html.Button('Create Project', id='project_button', n_clicks=0,
                                                    className="button_link"),
                               style={'width': '100%', 'padding-left': '87%', 'padding-right': '0%'}),
                      html.Br(),
                      html.Div(children=[html.A(
                          children=html.Button('Download Clinical Data template', id='download_button', n_clicks=0,
                                               style={'fontSize': '16px', 'display': 'block'}),
                          id='download_link', href='', n_clicks=0)],
                          style={'width': '100%', 'padding-left': '87%', 'padding-right': '0%'}),
                      html.Br(),
                      html.Div(children=[html.H1(id='project-creation')]),
                      html.Br()]),
            html.Hr()])]

        return project_creation_layout


def get_data():
    driver = connector.getGraphDatabaseConnectionConfiguration()
    if driver is not None:
        try:
            users = []
            tissues = []
            diseases = []
            user_nodes = connector.find_nodes(driver, node_type='User')
            tissue_nodes = connector.find_nodes(driver, node_type='Tissue')
            disease_nodes = connector.find_nodes(driver, node_type='Disease')
            for user in user_nodes:
                users.append((user['n']['name']))
            for tissue in tissue_nodes:
                tissues.append((tissue['n']['name']))
            for disease in disease_nodes:
                diseases.append((disease['n']['name']))
            return users, tissues, diseases
        except Exception as e:
            print(f"Error getting data: {e}")
            return None


def image_formatter(im):
    data_im = base64.b64encode(im).decode('ascii')
    return f'<img src="data:image/jpeg;base64,{data_im}">'


@dash.callback([Output('project-creation', 'children'),
                Output('update_project_id', 'children'),
                Output('update_project_id', 'style'),
                Output('download_button', 'style')],
               [Input('project_button', 'n_clicks')],
               [State('project name', 'value'),
                State('project acronym', 'value'),
                State('responsible-picker', 'value'),
                State('participant-picker', 'value'),
                State('data-types-picker', 'value'),
                State('number_timepoints', 'value'),
                State('related_to', 'value'),
                State('disease-picker', 'value'),
                State('tissue-picker', 'value'),
                State('intervention-picker', 'value'),
                # State('number_subjects', 'value'),
                State('project description', 'value'),
                State('date-picker-start', 'date'),
                State('date-picker-end', 'date')])
def create_project(n_clicks, name, acronym, responsible, participant, datatype, timepoints, related_to, disease, tissue,
                   intervention, description, start_date, end_date):
    config = builder_utils.setup_config('builder')
    separator = config["separator"]
    if n_clicks > 0:
        session_cookie = flask.request.cookies.get('custom-auth-session')
        responsible = separator.join(responsible)
        participant = separator.join(participant)
        datatype = separator.join(datatype)
        disease = separator.join(disease)
        tissue = separator.join(tissue)
        arguments = [name, datatype, disease, tissue, responsible]
        driver = connector.getGraphDatabaseConnectionConfiguration()

        if driver is not None:
            # Check if clinical variables exist in the database
            if intervention is not None:
                intervention = intervention.strip()
                if intervention != '':
                    interventions = list()
                    exist = dict()
                    for i in intervention.split(separator):
                        res = projectCreation.check_if_node_exists(driver, 'Clinical_variable', 'id', i)
                        if res.empty:
                            exist[i] = True
                        else:
                            exist[i] = False
                            interventions.append('{} ({})'.format(res['n.name'][0], i))
                    intervention = separator.join(interventions)

                    if any(exist.values()):
                        response = 'The intervention(s) "{}" specified does(do) not exist.'.format(
                            ', '.join([k for k, n in exist.items() if n == True]))
                        return response, None, {'display': 'none'}, {'display': 'none'}

            if any(not arguments[n] for n, i in enumerate(arguments)):
                response = "Insufficient information to create project. Fill in all fields with '*'."
                return response, None, {'display': 'none'}, {'display': 'none'}

            # Get project data from filled-in fields
            projectData = pd.DataFrame(
                [name, acronym, description, related_to, datatype, timepoints, disease, tissue, intervention,
                 responsible, participant, start_date, end_date]).T
            projectData.columns = ['name', 'acronym', 'description', 'related_to', 'datatypes', 'timepoints', 'disease',
                                   'tissue', 'intervention', 'responsible', 'participant', 'start_date', 'end_date']
            projectData['status'] = ''

            projectData.fillna(value=pd.np.nan, inplace=True)
            projectData.replace('', np.nan, inplace=True)

            # Generate project internal identifier bsed on timestamp
            # Excel file is saved in folder with internal id name
            epoch = time.time()
            internal_id = "%s%d" % ("CP", epoch)
            projectData.insert(loc=0, column='internal_id', value=internal_id)
            result = create_new_project.apply_async(args=[internal_id, projectData.to_json(), separator],
                                                    task_id='project_creation_' + session_cookie + internal_id,
                                                    queue='creation')
            result_output = result.get()
            if len(result_output) > 0:
                external_id = list(result_output.keys())[0]
                done_msg = result_output[external_id]
                if external_id != '' and done_msg is not None:
                    response = "Project successfully submitted. Download Clinical Data template."
                elif done_msg is None:
                    response = "There was a problem when creating the project. Please, contact the administrator."
                else:
                    response = 'A project with the same name already exists in the database.'
            else:
                response = "There was a problem when creating the project. Please, try again or contact the administrator."
                external_id = response
        else:
            response = "The Database is temporarily offline. Contact your administrator or start the datatabase."

        return response, '- ' + external_id, {'display': 'inline-block'}, {'display': 'block'}
    else:
        return None, None, {'display': 'none'}, {'display': 'none'}


@dash.callback(Output('project-creation', 'style'),
               [Input('project-creation', 'children')])
def change_style(style):
    if style is not None and 'successfully' in style:
        return {'fontSize': '20px', 'marginLeft': '70%', 'color': 'black'}
    else:
        return {'fontSize': '20px', 'marginLeft': '70%', 'color': 'red'}


@dash.callback(Output('download_link', 'href'),
               [Input('update_project_id', 'children')])
def update_download_link(project):
    if project is not None and project != '':
        return '/apps/templates{}'.format('Design_and_Clinical_templates')
    else:
        return ''
