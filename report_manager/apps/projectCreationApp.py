import os
import pandas as pd

import dash_core_components as dcc
import dash_html_components as html

from apps import basicApp

from graphdb_connector import connector


driver = connector.getGraphDatabaseConnectionConfiguration()

DataTypes = ['proteomics', 'clinical', 'wes', 'longitudinal_proteomics', 'longitudinal_clinical']
Users = [(u['name']) for u in driver.nodes.match("User")]
Tissues = [(t['name']) for t in driver.nodes.match("Tissue")]
Diseases = [(d['name']) for d in driver.nodes.match("Disease")]
#ClinicalVariables = [(c['name']) for c in driver.nodes.match("Clinical_variable")]
ClinicalVariables = pd.read_csv(os.path.join(os.getcwd(), 'apps/templates/tmp_data/clinicalvariables.csv'))

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
                    html.Div([html.Div(children=[html.A(children=html.Button('Download Template (.xlsx)', id='download_button', style={'maxWidth':'130px'}), id='download_link', style={'marginLeft': '90%'})]),
                              html.Div(id='dumm-div', style={'display':'none'}),
                              html.H4('Project information', style={'width':'15.5%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.H4('', id='update_project_id', style={'width':'15%', 'verticalAlign':'top', 'display':'none'}),


                              
                              html.Div(children=[html.Label('Project name:', style={'marginTop':15}),
                                                 dcc.Input(id='project name', placeholder='Insert name...', type='text', style={'width':'100%', 'height':'35px'})],
                                                 style={'width':'100%'}),
                              html.Div(children=[html.Label('Project Acronym:', style={'marginTop':15}),
                                                 dcc.Input(id='project acronym', placeholder='Insert name...', type='text', style={'width':'100%', 'height':'35px'})],
                                                 style={'width':'100%'}),
                              html.Div(children=[html.Label('Project Responsible:', style={'marginTop':15})],
                                                 style={'width':'49%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Label('Project Participants:', style={'marginTop':15})],
                                                 style={'width':'49%', 'marginLeft':'2%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='responsible-picker', options=[{'label':i, 'value':i} for i in Users], value=['',''], multi=True, searchable=True, style={'width':'100%'})],                                    
                                                 style={'width':'20%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Button('Add', id='add_responsible', style={'height':'35px'})],
                                                 style={'width':'10%', 'marginLeft': '0.4%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='participant-picker', options=[{'label':i, 'value':i} for i in Users], value=['',''], multi=True, searchable=True, style={'width':'100%'})],                                    
                                                 style={'width':'20%', 'marginLeft':'20.6%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Button('Add', id='add_participant', style={'height':'35px'})],
                                                 style={'width':'10%', 'marginLeft': '0.4%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Input(id='responsible', value='', type='text', disabled=True, style={'width':'100%', 'height':'35px', 'marginTop':5})],
                                                 style={'width':'49%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Input(id='participant', value='', type='text', disabled=True, style={'width':'100%', 'height':'35px', 'marginTop':5})],
                                                 style={'width':'49%', 'marginLeft':'2%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Label('Project Data Types:', style={'marginTop':10})],
                                                 style={'width':'49%', 'marginLeft':'0%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Label('Project Disease:', style={'marginTop':10})],
                                                 style={'width':'49%', 'marginLeft':'2%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='data-types-picker', options=[{'label':i, 'value':i} for i in DataTypes], value=['',''], multi=True, searchable=True, style={'width':'100%'})],
                                                 style={'width':'20%', 'marginLeft': '0%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Button('Add', id='add_datatype', style={'height':'35px'})],
                                                 style={'width':'10%', 'marginLeft': '0.4%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='disease-picker', options=[{'label':i, 'value':i} for i in Diseases], value=['',''], multi=True, searchable=True, style={'width':'100%'})],                                    
                                                 style={'width':'20%', 'marginLeft':'20.6%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Button('Add', id='add_disease', style={'height':'35px'})],
                                                 style={'width':'10%', 'marginLeft': '0.4%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Input(id='data-types', value='', type='text', disabled=True, style={'width':'100%', 'height':'35px', 'marginTop':5})],
                                                 style={'width':'49%', 'marginLeft':'0%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Input(id='disease', value='', type='text', disabled=True, style={'width':'100%', 'height':'35px', 'marginTop':5})],
                                                 style={'width':'49%', 'marginLeft':'2%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Label('Project Tissue:', style={'marginTop':10})],
                                                 style={'width':'49%', 'marginLeft':'0%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Label('Project Intervention:', style={'marginTop':10})],
                                                 style={'width':'49%', 'marginLeft':'2%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='tissue-picker', options=[{'label':i, 'value':i} for i in Tissues], value=['',''], multi=True, searchable=True, style={'width':'100%'})],
                                                 style={'width':'20%', 'marginLeft': '0%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Button('Add', id='add_tissue', style={'height':'35px'})],
                                                 style={'width':'10%', 'marginLeft': '0.4%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='intervention-picker', options=[{'label':i, 'value':i} for i in ClinicalVariables['n.name']], value=['',''], multi=True, searchable=True, style={'width':'100%'})],
                                                 style={'width':'20%', 'marginLeft': '20.6%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Button('Add', id='add_intervention', style={'height':'35px'})],
                                                 style={'width':'10%', 'marginLeft': '0.4%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Input(id='tissue', value='', type='text', disabled=True, style={'width':'100%', 'height':'35px', 'marginTop':5})],
                                                 style={'width':'49%', 'marginLeft':'0%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Input(id='intervention', value='', type='text', disabled=False, style={'width':'100%', 'height':'35px', 'marginTop':5})],
                                                 style={'width':'49%', 'marginLeft':'2%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Label('Number of subjects:', style={'marginTop':15})],
                                                 style={'width':'49%', 'marginLeft':'0%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[html.Label('Timepoints:', style={'marginTop':15})],
                                                 style={'width':'49%', 'marginLeft':'2%', 'verticalAlign':'top', 'display':'inline-block'}),
                              html.Div(children=[dcc.Input(id='number_subjects', placeholder='E.g. 77 (each unique patient counts as 1 subject)', type='text', style={'width':'100%', 'height':'35px'})],
                                                 style={'width':'49%', 'marginLeft':'0%', 'verticalAlign':'top', 'display':'inline-block'}),                              
                              html.Div(children=[dcc.Input(id='number_timepoints', placeholder='E.g. 2 months, 15 days, 24 hours...', type='text', style={'width':'100%', 'height':'35px'})],
                                                 style={'width':'49%', 'marginLeft':'2%', 'verticalAlign':'top', 'display':'inline-block'}),            
                              html.Div(children=[html.Label('Project Description:', style={'marginTop':15}),
                                                 dcc.Textarea(id='project description', placeholder='Enter description...', style={'width':'100%', 'height':'100px'})]),
                              html.Div(children=[html.Label('Starting Date:', style={'marginTop':10}),
                                                 dcc.DatePickerSingle(id='date-picker-start', placeholder='Select date...', clearable=True)],
                                                 style={'width':'30%', 'verticalAlign':'top', 'marginTop':10, 'display':'inline-block'}),
                              html.Div(children=[html.Label('Ending Date:', style={'marginTop':10}),
                                                 dcc.DatePickerSingle(id='date-picker-end', placeholder='Select date...', clearable=True)],
                                                 style={'width':'30%', 'verticalAlign':'top', 'marginTop':10, 'display':'inline-block'}),
                              html.Div(children=[html.Button('Create Project', id='project_button')],
                                                 style={'fontSize':'22px', 'marginLeft':'87.3%'}),
                              html.Div(id='project-creation', style={'fontSize':'20px', 'marginLeft':'70%'})]),
                    html.Hr()])]

        self.extend_layout(layout)
