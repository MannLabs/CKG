import io
import os
import pandas as pd
import base64
import flask

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash_network import Network

from app import app
from apps import initialApp, projectApp, importsApp, projectCreationApp

template_cols = pd.read_excel(os.path.join(os.getcwd(), 'apps/templates/ClinicalData_template.xlsx'))
template_cols = template_cols.columns.tolist()

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content', style={'padding-top':50}),
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname is not None:
        if pathname == '/apps/initial' or pathname == '/':
            return initialApp.layout

        elif pathname.startswith('/apps/projectCreation'):
            projectCreation = projectCreationApp.ProjectCreationApp("Project Creation", "", "", layout = [], logo = None, footer = None)
            return projectCreation.layout
        elif pathname.startswith('/apps/project'):
            projectId = pathname.split('/')[-1]
            project = projectApp.ProjectApp(projectId, projectId, "", "", layout = [], logo = None, footer = None)
            return project.layout
        elif pathname.startswith('/apps/imports'):
            imports = importsApp.ImportsApp("CKG imports monitoring", "Statistics", "", layout = [], logo = None, footer = None)
            return imports.layout
        else:
            return '404'

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
    if n_clicks is not None and n_clicks > 0:
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
            for i in df.columns:
                if i not in template_cols:
                    existing_columns.append({'id': i, 'name': i,
                                             'editable_name': False, 'deletable': True})
                else:
                    pass

        if n_clicks is not None and n_clicks > 0:
            for j in value:
                existing_columns.append({'id': j, 'name': j,
                                     'editable_name': False, 'deletable': True})
        return data, existing_columns



if __name__ == '__main__':
    app.run_server(debug=True, port=5000)
