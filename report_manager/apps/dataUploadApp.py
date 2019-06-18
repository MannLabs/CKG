import base64
import io
import os
import pandas as pd
import json
import py2neo

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import flask

import config.ckg_config as ckg_config
from apps import basicApp

from graphdb_connector import connector
import logging
import logging.config


driver = connector.getGraphDatabaseConnectionConfiguration()

DataTypes = ['proteomics', 'clinical', 'wes', 'longitudinal_proteomics', 'longitudinal_clinical']
Tissues = [(t['name']) for t in driver.nodes.match("Tissue")]
Diseases = [(d['name']) for d in driver.nodes.match("Disease")]

query = 'MATCH (n:Clinical_variable) RETURN n.name,n.id LIMIT 20'
df = pd.DataFrame(connector.getCursorData(driver, query).values)
df[0] = ['({0})'.format(i) for i in df[0].tolist()]
ClinicalVariables = df[[1, 0]].apply(lambda x: ' '.join(x),axis=1).tolist()

# template_cols = pd.read_excel(os.path.join(os.getcwd(), 'apps/templates/ClinicalData_template.xlsx'))
# template_cols = template_cols.columns.tolist()


class DataUploadApp(basicApp.BasicApp):
    def __init__(self, projectId, title, subtitle, description, layout = [], logo = None, footer = None):
        self._project_id = projectId
        self.pageType = "UploadDataPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    @property
    def project_id(self):
        return self._project_id

    @project_id.setter
    def project_id(self, project_id):
        self._project_id = project_id

    def buildPage(self):
        projectID = self.project_id
        self.add_basic_layout()
        layout = [html.Div([
                            html.Div([html.H4('Upload Experiment file', style={'marginTop':30, 'marginBottom':20}),
                                      dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                                                 style={'width': '100%',
                                                        'height': '60px',
                                                        'lineHeight': '60px',
                                                        'borderWidth': '1px',
                                                        'borderStyle': 'dashed',
                                                        'borderRadius': '5px',
                                                        'textAlign': 'center',
                                                        'margin': '0px'},
                                                 multiple=False)]),
                            html.Br(),
        					html.Div(children=[html.Label('Select upload data type:', style={'marginTop':10})],
                                               style={'width':'49%', 'marginLeft':'0%', 'verticalAlign':'top', 'fontSize':'18px'}),
                            html.Div(id='dumm-div', style={'display':'none'}),
							html.Div(children=[dcc.Dropdown(id='upload-data-type-picker', options=[{'label':i, 'value':i} for i in DataTypes], value='', multi=False, style={'width':'100%'})],
                                               style={'width':'20%', 'marginLeft': '0%', 'verticalAlign':'top', 'display':'inline-block'}),
                            html.Div(children=[html.Button('Add', id='add_upload_datatype', style={'height':'35px'})],
                                               style={'width':'10%', 'marginLeft': '0.4%', 'verticalAlign':'top', 'display':'inline-block'}),
							html.Div(children=[dcc.Input(id='upload-data-type', value='', type='text', style={'width':'100%', 'height':'35px', 'marginTop':5})],
                                                 style={'width':'49%', 'marginLeft':'0%', 'verticalAlign':'top'}),
                			html.Div([
                    				  html.Div(children=[html.A(children=html.Button('Export Table', id='data_download_button', style={'height':36, 'maxWidth':'200px'}), id='data_download_link', download='downloaded_ClinicalData.csv')],
                                      					 style={'marginTop':'0%', 'marginLeft': '91.4%', 'horizontalAlign':'right', 'verticalAlign':'top', 'display':'inline-block'}),
                    				  html.Div(children=[dcc.Dropdown(id='clinical-variables-picker', placeholder='Select clinical variables...', options=[{'label':i, 'value':i} for i in ClinicalVariables], value=['', ''], multi=True, style={})],
                             							 style={'marginTop':'1%', 'width':'25%', 'verticalAlign':'top', 'display':'inline-block'}),
                    				  html.Div(children=[html.Button('Add Column', id='editing-columns-button', n_clicks=0, style={'height':36, 'width':100})],
                             							 style={'marginTop':'1%', 'marginLeft': 5, 'verticalAlign':'top', 'display':'inline-block'}),
                    				  html.Div(children=[dcc.Dropdown(id='tissue-finder', placeholder='Search for tissue name...', options=[{'label':i, 'value':i} for i in Tissues], value=[''], multi=True, clearable=False)],
                             							 style={'marginTop':'1%', 'width':'15%', 'marginLeft': '20.3%', 'verticalAlign':'top', 'display':'inline-block'}),
                    				  html.Div(children=[dcc.Dropdown(id='disease-finder', placeholder='Search for disease name...', options=[{'label':i, 'value':i} for i in Diseases], value=[''], multi=True, clearable=False)],
                             							 style={'marginTop':'1%', 'width':'15%', 'marginLeft': 5, 'verticalAlign':'top', 'display':'inline-block'}),
                    				  html.Div(children=[dcc.Dropdown(id='intervention-finder', placeholder='Search for intervention name...', options=[{'label':i, 'value':i} for i in Diseases], value=[''], multi=True, clearable=False)],
                             							 style={'marginTop':'1%', 'width':'15%', 'marginLeft': 5, 'verticalAlign':'top', 'display':'inline-block'}),
                    				  html.Div([dash_table.DataTable(id='clinical-table', columns=[{"name": '', "id": '', 'editable_name':False, 'deletable':True} for i in range(5)],
                                      		                         data=[{j: ''} for j in range(10)], n_fixed_rows=1, editable=True, style_cell={'minWidth': '220px', 'width': '220px','maxWidth': '220px', 'whiteSpace': 'normal'},
                                                                     css=[{'selector': '.dash-cell div.dash-cell-value', 'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}])],
                             					style={'marginTop':'1%', 'maxHeight': 800, 'overflowY': 'scroll', 'overflowX': 'scroll'}),
                					  html.Div(children=[html.Button('Upload Data', id='submit_button')],
                             							 style={'marginTop':'5%', 'fontSize':'22px', 'minWidth':'500px', 'marginLeft':'89%'})]),
                            html.Div(id='data-upload', style={'fontSize':'20px', 'marginLeft':'70%'}),
                			html.Hr()])]


        self.extend_layout(layout)
