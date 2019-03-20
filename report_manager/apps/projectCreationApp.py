import base64
import datetime
import io
import os
import pandas as pd
import xlsxwriter
import csv
import json
import uuid

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import dash_table
import flask


import report_manager.app
from report_manager.app import app

###Create queries for these
Users = pd.read_csv('/Users/plh450/Clinical_Proteomics/CKG/ClinicalDataApp/CKG_users.csv')
DataTypes = pd.read_csv('/Users/plh450/Clinical_Proteomics/CKG/ClinicalDataApp/CKG_datatypes.csv')
Tissues = pd.read_csv('/Users/plh450/Clinical_Proteomics/CKG/ClinicalDataApp/TissueNames.csv')
ClinicalVariables = pd.read_csv('/Users/plh450/Clinical_Proteomics/CKG/ClinicalDataApp/clinicalvariables.csv')
###Works
template_cols = pd.read_excel(os.path.join(os.getcwd(), 'apps/templates/ClinicalData_template.xlsx'))
template_cols = template_cols.columns.tolist()


app.layout = html.Div([
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
                # html.Div([html.H5('Experiment information', style={'marginTop':30, 'display':'none'}),
                #     html.Div(children=[html.Label('Subject identifiers:', style={'marginTop':20}),
                #                        dcc.Textarea(id='subject ids', placeholder='Insert subject identifiers list', style={'width':'100%'})],
                #                        style={'display':'none'}),
                #     html.Div(children=[html.Label('Biological sample identifiers:', style={'marginTop':10}),
                #                        dcc.Textarea(id='biosample ids', placeholder='Insert biological sample identifiers list', style={'width':'100%'})],
                #                        style={'display':'none'}),
                #     html.Div(children=[html.Label('Analytical sample identifiers:', style={'marginTop':10}),
                #                        dcc.Textarea(id='analyticsample ids', placeholder='Insert analytical sample identifiers list', style={'width':'100%'})],
                #                        style={'display':'none'}),
                #     html.Div(children=[html.Label('Biological sample tissues:', style={'marginTop':10}),
                #                        dcc.Textarea(id='biosample tissues', placeholder='Insert biological sample timepoints list', style={'width':'100%'})],
                #                        style={'display':'none'}),
                #     html.Div(children=[html.Label('Biological sample diseases:', style={'marginTop':10}),
                #                        dcc.Textarea(id='biosample diseases', placeholder='Insert biological sample disease list', style={'width':'100%'})],
                #                        style={'display':'none'}),
                #     html.Div(children=[html.Label('Biological sample interventions:', style={'marginTop':10}),
                #                        dcc.Textarea(id='biosample interventions', placeholder='Insert biological sample intervention list', style={'width':'100%'})],
                #                        style={'display':'none'}),
                #     html.Div(children=[html.Label('Biological sample timepoints and units:', style={'marginTop':10}),
                #                        dcc.Textarea(id='biosample timepoints', placeholder='Insert biological sample timepoints list', style={'width':'100%'})],
                #                        style={'display':'none'}),
                #     html.Div(children=[html.Label('Biological sample groups:', style={'marginTop':10}),
                #                        dcc.Textarea(id='grouping1', placeholder='Insert biological sample grouping list', style={'width':'100%'})],
                #                        style={'display':'none'}),
                #     html.Div(children=[html.Label('Biological sample groups (secondary):', style={'marginTop':10}),
                #                        dcc.Textarea(id='grouping2', placeholder='Insert biological sample grouping list', style={'width':'100%'})],
                #                        style={'display':'none'})]),
                html.Div([html.H4('Clinical Data', style={'marginTop':20}),
                    html.Div(children=[dcc.Dropdown(id='clinical-variables-picker', placeholder='Select clinical variables...', options=[{'label':i, 'value':i} for i in ClinicalVariables['n.name']], value=['', ''], multi=True, style={})],
                             style={'marginTop':20, 'width':'30%', 'verticalAlign':'top', 'display':'inline-block'}),
                    html.Div(children=[html.Button('Add Column', id='editing-columns-button', n_clicks=0, style={'height':36, 'width':100})],
                             style={'marginTop':20, 'marginLeft': 5, 'verticalAlign':'top', 'display':'inline-block'}),
                    html.Div([dash_table.DataTable(id='clinical-table', columns=[{"name": i, "id": i, 'editable_name':False, 'deletable':True} for i in template_cols],
                                                            data=[{i: '' for i in template_cols} for j in range(10)], editable=True),
                              html.Hr()],
                             style={'marginTop':15, 'maxHeight': 800, 'overflowY': 'scroll', 'overflowX': 'scroll'})])])



def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' or 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return None
    return df


@app.callback(Output('download_link', 'href'),
             [Input('download_button', 'n_clicks')])
def update_download_link(n_clicks):
    relative_filename = os.path.join('apps/templates', 'ClinicalData_template.xlsx')
    absolute_filename = os.path.join(os.getcwd(), relative_filename)
    if n_clicks > 0:
        return '/{}'.format(relative_filename)


@app.server.route('/apps/templates/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'apps/templates'), path)


@app.callback([Output('clinical-table', 'data'),
               Output('clinical-table', 'columns')],
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('editing-columns-button', 'n_clicks')],
              [State('clinical-variables-picker', 'value'),
               State('clinical-table', 'columns')])
def update_data(contents, filename, n_clicks, value, existing_columns):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            data = df.to_dict('rows')

        if n_clicks > 0:
            for i in value:
                existing_columns.append({'id': i, 'name': i,
                                     'editable_name': False, 'deletable': True})
        return data, existing_columns



if __name__ == '__main__':
    app.run_server(debug=True)
