import io
import os
import pandas as pd


import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import flask

import config.ckg_config as ckg_config
from apps import basicApp


###Create queries for these
Users = pd.read_csv('/Users/plh450/Clinical_Proteomics/CKG/ClinicalDataApp/CKG_users.csv')
DataTypes = pd.read_csv('/Users/plh450/Clinical_Proteomics/CKG/ClinicalDataApp/CKG_datatypes.csv')
Tissues = pd.read_csv('/Users/plh450/Clinical_Proteomics/CKG/ClinicalDataApp/TissueNames.csv')
ClinicalVariables = pd.read_csv('/Users/plh450/Clinical_Proteomics/CKG/ClinicalDataApp/clinicalvariables.csv')
###Works
template_cols = pd.read_excel(os.path.join(os.getcwd(), 'apps/templates/ClinicalData_template.xlsx'))
template_cols = template_cols.columns.tolist()



class ProjectCreationApp(basicApp.BasicApp):
    def __init__(self, title, subtitle, description, layout = [], logo = None, footer = None):
        self.pageType = "projectCreationPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        self.add_basic_layout()
        layout = [html.Div([
                    html.Div([html.Div(children=[html.A(children=html.Button('Download Template (.xlsx)', id='download_button', style={'maxWidth':'130px'}), id='download_link', style={'marginLeft': '90%',})]),
                    html.H4('Project information', style={'marginTop':0}),
                    html.Div(children=[html.Label('Project name:', style={'marginTop':10}),
                                       dcc.Input(id='project name', placeholder='Insert name...', type='text', style={'width':'100%'})],
                                       style={'width':'100%'}),
                    html.Div(children=[html.Label('Project Acronym:', style={'marginTop':10}),
                                       dcc.Input(id='project acronym', placeholder='Insert name...', type='text', style={'width':'100%'})],
                                       style={'width':'100%'}),
                    html.Div(children=[html.Label('Project Responsible:', style={'marginTop':10}),
                                       dcc.Dropdown(id='responsible-picker', options=[{'label':i, 'value':i} for i in Users.name], value='', multi=True, style={'width':'100%'})],
                                       style={'width':'49%', 'verticalAlign':'top', 'display':'inline-block'}),
                    html.Div(children=[html.Label('Project Data Types:', style={'marginTop':10}),
                                       dcc.Dropdown(id='data-types-picker', options=[{'label':i, 'value':i} for i in DataTypes.name], value='', multi=True, style={'width':'100%'})],
                                       style={'width':'49%', 'verticalAlign':'top', 'marginLeft':'2%', 'display':'inline-block'}),
                    html.Div(children=[html.Label('Project Participants:', style={'marginTop':10}),
                                       dcc.Dropdown(id='participant-picker', options=[{'label':i, 'value':i} for i in Users.name], value='', multi=True, style={'width':'100%'})],
                                       style={'width':'49%', 'verticalAlign':'top', 'marginLeft':'0%', 'display':'inline-block'}),
                    html.Div(children=[html.Label('Project Tissue:', style={'marginTop':10}),
                                       dcc.Dropdown(id='tissue-picker', options=[{'label':i, 'value':i} for i in Tissues['n.name']], value='', multi=True, style={'width':'100%'})],
                                       style={'width':'49%', 'verticalAlign':'top', 'marginLeft':'2%', 'display':'inline-block'}),
                    html.Div(children=[html.Label('Project Description:', style={'marginTop':10}),
                                       dcc.Textarea(id='project description', placeholder='Enter description...', style={'width':'100%'})]),
                    html.Div(children=[html.Label('Starting Date:', style={'marginTop':10}),
                                       dcc.DatePickerSingle(id='date-picker-start', placeholder='Select date...', clearable=True)],
                                       style={'width':'30%', 'verticalAlign':'top', 'marginTop':10, 'display':'inline-block'}),
                    html.Div(children=[html.Label('Ending Date:', style={'marginTop':10}),
                                       dcc.DatePickerSingle(id='date-picker-end', placeholder='Select date...', clearable=True)],
                                       style={'width':'30%', 'verticalAlign':'top', 'marginTop':10, 'display':'inline-block'}),
                    html.Br()]),
                html.Div([html.H4('Upload Experiment file', style={'marginTop':30, 'marginBottom':20}),
                          dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                                     style={'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'},
                                     multiple=False),
                          html.Br()]),
                html.Div([html.H4('Clinical Data', style={'marginTop':20}),
                    html.Div(children=[dcc.Dropdown(id='clinical-variables-picker', placeholder='Select clinical variables...', options=[{'label':i, 'value':i} for i in ClinicalVariables['n.name']], value=['', ''], multi=True, style={})],
                             style={'marginTop':20, 'width':'30%', 'verticalAlign':'top', 'display':'inline-block'}),
                    html.Div(children=[html.Button('Add Column', id='editing-columns-button', n_clicks=0, style={'height':36, 'width':100})],
                             style={'marginTop':20, 'marginLeft': 5, 'verticalAlign':'top', 'display':'inline-block'}),
                    html.Div([dash_table.DataTable(id='clinical-table', columns=[{"name": i, "id": i, 'editable_name':False, 'deletable':True} for i in template_cols],
                                                            data=[{i: '' for i in template_cols} for j in range(10)], editable=True),
                              html.Hr()],
                             style={'marginTop':15, 'maxHeight': 800, 'overflowY': 'scroll', 'overflowX': 'scroll'})])])]

        self.extend_layout(layout)