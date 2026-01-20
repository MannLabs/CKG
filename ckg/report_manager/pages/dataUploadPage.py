import os
import re
import shutil
from datetime import datetime

import dash
import flask
import pandas as pd
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ckg import ckg_utils
from ckg.graphdb_builder import builder_utils
from ckg.graphdb_builder.builder import loader
from ckg.graphdb_builder.experiments import experiments_controller as eh
from ckg.graphdb_connector import connector
from ckg.report_manager import utils
from ckg.report_manager.apps import dataUpload
from ckg.report_manager.worker import create_new_identifiers

ckg_config = ckg_utils.read_ckg_config()
config = builder_utils.setup_config('builder')
separator = config["separator"]

title = "Data Upload"
subtitle = ""
description = ""

dash.register_page(__name__, path='/apps/dataUploadApp', title=f"{title} - {subtitle}", description=description)

DataTypes = ['experimental_design', 'clinical', 'proteomics', 'interactomics', 'phosphoproteomics']


def layout():
    session_cookie = flask.request.cookies.get('custom-auth-session')
    logged_in = session_cookie is not None
    if logged_in == False:
        return html.Div(["Please ", dcc.Link("login", href="/apps/loginPage"), " to continue"])

    data_upload_layout = [html.Div([
        html.H1(children=title),
        html.H2(children=subtitle),
        html.Div(children=description),
        html.Div([html.H4('Project identifier:', style={'marginTop': 30, 'marginBottom': 20}),
                  dcc.Input(id='project_id', placeholder='e.g. P0000001', type='text', value='', debounce=True,
                            maxLength=8, minLength=8, style={'width': '100%', 'height': '55px'}),
                  dcc.Markdown(id='existing-project')],
                 style={'width': '20%'}),
        html.Br(),
        html.Div(id='upload-form', children=[
            html.Div(children=[html.A("Download example files",
                                      id='example_files',
                                      href='/example_files',
                                      n_clicks=0,
                                      className="button_link")],
                     style={'width': '100%', 'padding-left': '87%', 'padding-right': '0%'}),
            html.Div(children=[html.Label('Select upload data type:', style={'marginTop': 10})],
                     style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top', 'fontSize': '18px'}),
            html.Div(children=[
                dcc.RadioItems(id='upload-data-type-picker', options=[{'label': i, 'value': i} for i in DataTypes],
                               value=None,
                               inputStyle={"margin-right": "5px"}, style={'display': 'block', 'fontSize': '16px'})]),
            html.Div(children=[html.H5('Proteomics tool:'), dcc.RadioItems(id='prot-tool',
                                                                           options=[{'label': i, 'value': i} for i in
                                                                                    ['MaxQuant', 'DIA-NN',
                                                                                     'Spectronaut', 'FragPipe',
                                                                                     'mzTab']], value='',
                                                                           inputStyle={"margin-right": "5px"},
                                                                           style={'display': 'block',
                                                                                  'fontSize': '16px'})],
                     id='proteomics-tool', style={'padding-top': 20}),
            html.Div(children=[html.H5('Select the type of file uploaded:'), dcc.Dropdown(id='prot-file', options=[
                {'label': i, 'value': i} for i in ['Protein groups', 'Peptides', 'Phospho STY sites']], value='',
                                                                                          style={'display': 'block',
                                                                                                 'fontSize': '14px',
                                                                                                 'width': '250px'})],
                     id='proteomics-file', style={'padding-top': 20}),
            html.Div([html.H4('Upload file (max. 100Mb)', style={'marginTop': 30, 'marginBottom': 20}),
                      dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                                 style={'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '0px'},
                                 multiple=False, max_size=1024 * 1024 * 1000)]),
            html.Br(),
            html.Div(
                children=[dcc.Markdown('**Uploaded Files:**', id='markdown-title'), dcc.Markdown(id='uploaded-files')]),
            html.Div([html.Button("Upload Data to CKG",
                                  id='submit_button',
                                  n_clicks=0,
                                  className="button_link")],
                     style={'width': '100%', 'padding-left': '87%', 'padding-right': '0%'})]),
        html.Div(children=[
            html.A('Download Files(.zip)',
                   id='data_download_link',
                   href='',
                   n_clicks=0,
                   style={'display': 'none'},
                   className="button_link")]),
        html.Div(children=[
            html.A(children='',
                   id='link-project-report',
                   href='',
                   target='',
                   n_clicks=0,
                   style={'display': 'none'},
                   className="button_link")]),
        html.Div(id='data-upload-result', children=[dcc.Markdown(id='upload-result')], style={'width': '100%'}),
        html.Hr()]),
        html.Div(id='project_table', children=[])]
    return data_upload_layout


@dash.callback([Output('existing-project', 'children'),
                Output('upload-form', 'style'),
                Output('link-project-report', 'children'),
                Output('link-project-report', 'href')],
               [Input('project_id', 'value')],
               [State('data_download_link', 'style')])
def activate_upload_form(projectid, download_style):
    m = ''
    style = {'pointer-events': 'none', 'opacity': 0.5}
    download_style.update({'display': 'none'})
    report_title = ''
    report_href = ''
    driver = connector.getGraphDatabaseConnectionConfiguration()
    if driver is not None:
        if len(projectid) > 7:
            project = connector.find_node(driver, node_type='Project', parameters={'id': projectid})
            if len(project) == 0:
                m = 'ERROR: Project "{}" does not exist in the database.'.format(projectid)
            else:
                if 'name' in project:
                    report_title = 'Generate report: {}'.format(project['name'])
                    report_href = '/apps/project?project_id={}&force=0'.format(projectid)
                    m = 'Uploading data for Project: **{}**'.format(project['name'])
                style = {}
    else:
        m = 'ERROR: Database if temporarily offline. Contact your administrator or start the database.'

    return m, style, report_title, report_href


@dash.callback(Output('proteomics-tool', 'style'),
               [Input('upload-data-type-picker', 'value'),
                Input('prot-tool', 'value')])
def show_proteomics_options(datatype, prot_tool):
    display = {'display': 'none'}
    if datatype in ['proteomics', 'interactomics', 'phosphoproteomics']:
        if prot_tool == '':
            display = {'display': 'block'}
        else:
            display = {'display': 'block'}

    return display


@dash.callback([Output('proteomics-file', 'style'),
                Output('upload-data', 'disabled')],
               [Input('upload-data-type-picker', 'value'),
                Input('prot-tool', 'value'),
                Input('prot-file', 'value')])
def show_proteomics_file_options(datatype, prot_tool, prot_file):
    display = ({'display': 'none'}, False)
    if datatype in ['proteomics', 'interactomics', 'phosphoproteomics']:
        if prot_tool is not None and prot_tool != '':
            if prot_file == '' and prot_tool != 'mzTab':
                display = ({'display': 'block'}, True)
            else:
                display = ({'display': 'block'}, False)
        else:
            display = ({'display': 'block'}, True)

    return display


@dash.callback([Output('uploaded-files', 'children'),
                Output('upload-data', 'filename'),
                Output('prot-tool', 'value'),
                Output('prot-file', 'value')],
               [Input('upload-data', 'contents')],
               [State('upload-data-type-picker', 'value'),
                State('prot-tool', 'value'),
                State('prot-file', 'value'),
                State('project_id', 'value'),
                State('upload-data', 'filename')])
def save_files_in_tmp(content, dataset, prot_tool, prot_file, projectid, uploaded_file):
    if dataset is not None:
        session_cookie = flask.request.cookies.get('custom-auth-session')
        temporaryDirectory = os.path.join(ckg_config['tmp_directory'], session_cookie + "upload")
        if not os.path.exists(ckg_config['tmp_directory']):
            os.makedirs(ckg_config['tmp_directory'])
        elif not os.path.exists(temporaryDirectory):
            os.makedirs(temporaryDirectory)

        directory = os.path.join(temporaryDirectory, dataset)
        if os.path.exists(directory) and uploaded_file is not None:
            if os.path.exists(os.path.join(directory, uploaded_file)):
                shutil.rmtree(directory)

        builder_utils.checkDirectory(directory)
        if dataset in ['proteomics', 'interactomics', 'phosphoproteomics'] and prot_tool != '' and (
                prot_file != '' or prot_tool == 'mzTab'):
            selected_file = prot_tool.lower() + "-" + prot_file.lower()
            if selected_file in config['file_proteomics']:
                filename = config['file_proteomics'][selected_file]
            else:
                if prot_tool == 'mzTab':
                    filename = dataset + '_' + prot_tool.lower() + '.mztab'
                else:
                    filename = dataset + '_' + prot_tool.lower() + '_' + prot_file.replace(' ', '').lower() + '.' + \
                               uploaded_file.split('.')[-1]
            directory = os.path.join(directory, prot_tool.lower())
            if os.path.exists(directory):
                if os.path.exists(os.path.join(directory, filename)):
                    os.remove(os.path.join(directory, filename))
            builder_utils.checkDirectory(directory)
        elif dataset == 'experimental_design':
            filename = config['file_design'].split('_')[0] + '_' + projectid + '.' + uploaded_file.split('.')[-1]
        elif dataset == 'clinical':
            filename = config['file_clinical'].split('_')[0] + '_' + projectid + '.' + uploaded_file.split('.')[-1]

        if uploaded_file is None:
            content = None
        if content is not None:
            data = builder_utils.parse_contents(content, filename)
            builder_utils.export_contents(data, directory, filename)

            uploaded = uploaded_file
            uploaded_file = None
            return uploaded, uploaded_file, '', ''
        else:
            raise PreventUpdate

    return '', None, '', ''


@dash.callback([Output('upload-result', 'children'),
                Output('data_download_link', 'style'),
                Output('link-project-report', 'style'),
                Output('project_table', 'children')],
               [Input('submit_button', 'n_clicks'),
                Input('project_id', 'value')])
def run_processing(n_clicks, project_id):
    message = None
    style = {'display': 'none'}
    table = None

    if n_clicks > 0:
        session_cookie = flask.request.cookies.get('custom-auth-session')
        destDir = os.path.join(ckg_config['experiments_directory'], project_id)
        builder_utils.checkDirectory(destDir)
        temporaryDirectory = os.path.join(ckg_config['tmp_directory'], session_cookie + "upload")
        datasets = builder_utils.listDirectoryFoldersNotEmpty(temporaryDirectory)
        driver = connector.getGraphDatabaseConnectionConfiguration()
        if driver is not None:
            res_n = dataUpload.check_samples_in_project(driver, project_id)
            if 'experimental_design' in datasets:
                dataset = 'experimental_design'
                directory = os.path.join(temporaryDirectory, dataset)
                destination = os.path.join(destDir, dataset)
                experimental_files = os.listdir(directory)
                regex = r"{}.+".format(config['file_design'].replace('PROJECTID', project_id))
                r = re.compile(regex)
                experimental_filename = list(filter(r.match, experimental_files))
                if len(experimental_filename) > 0:
                    experimental_filename = experimental_filename.pop()
                    designData = builder_utils.readDataset(os.path.join(directory, experimental_filename))
                    designData = designData.astype(str)
                    designData.columns = [c.lower() for c in designData.columns]
                    if 'subject external_id' in designData.columns and 'biological_sample external_id' in designData.columns and 'analytical_sample external_id' in designData.columns:
                        if (res_n > 0).any().values.sum() > 0:
                            res = dataUpload.remove_samples_nodes_db(driver, project_id)
                            res_n = dataUpload.check_samples_in_project(driver, project_id)
                            if (res_n > 0).any().values.sum() > 0:
                                message = 'ERROR: There is already an experimental design loaded into the database and there was an error when trying to delete it. Contact your administrator.'
                                return message, style, style, table

                        res_n = None
                        result = create_new_identifiers.apply_async(
                            args=[project_id, designData.to_json(), directory, experimental_filename],
                            task_id='data_upload_' + session_cookie + datetime.now().strftime('%Y%m-%d%H-%M%S-'),
                            queue='creation')
                        result_output = result.wait(timeout=None, propagate=True, interval=0.2)
                        res_n = pd.DataFrame.from_dict(result_output['res_n'])
                        builder_utils.copytree(directory, destination)
                    else:
                        message = 'ERROR: The Experimental design file provided ({}) is missing some of the required fields: {}'.format(
                            experimental_filename, ','.join(['subject external_id', 'biological_sample external_id',
                                                             'analytical_sample external_id']))
                        builder_utils.remove_directory(directory)

                        return message, style, style, table

            if 'clinical' in datasets:
                dataset = 'clinical'
                directory = os.path.join(temporaryDirectory, dataset)
                clinical_files = os.listdir(directory)
                regex = r"{}.+".format(config['file_clinical'].replace('PROJECTID', project_id))
                r = re.compile(regex)
                clinical_filename = list(filter(r.match, clinical_files))
                if len(clinical_filename) > 0:
                    clinical_filename = clinical_filename.pop()
                    data = builder_utils.readDataset(os.path.join(directory, clinical_filename))
                    data.columns = [c.lower() for c in data.columns]
                    external_ids = {}
                    if 'subject external_id' in data and 'biological_sample external_id' in data:
                        external_ids['subjects'] = data['subject external_id'].astype(str).unique().tolist()
                        external_ids['biological_samples'] = data['biological_sample external_id'].astype(
                            str).unique().tolist()
                        dataUpload.create_mapping_cols_clinical(driver, data, directory, clinical_filename,
                                                                separator=separator)
                        if 0 in res_n.values:
                            samples = ', '.join([k for (k, v) in res_n if v == 0])
                            message = 'ERROR: No {} for project {} in the database. Please upload first the experimental design (ExperimentalDesign_{}.xlsx)'.format(
                                samples, project_id, project_id)
                            builder_utils.remove_directory(directory)

                            return message, style, style, table
                        else:
                            db_ids = dataUpload.check_external_ids_in_db(driver, project_id).to_dict()
                            message = ''
                            intersections = {}
                            differences_in = {}
                            differences_out = {}
                            for col in external_ids:
                                intersect = list(set(db_ids[col].values()).intersection(external_ids[col]))
                                difference_in = list(set(db_ids[col].values()).difference(external_ids[col]))
                                difference_out = list(set(external_ids[col]).difference(set(db_ids[col].values())))
                                if len(difference_in) > 0 or len(difference_out) > 0:
                                    intersections[col] = intersect
                                    differences_in[col] = difference_in
                                    differences_out[col] = difference_out
                            for col in intersections:
                                message += 'WARNING: Some {} identifiers were not matched:\n Matching: {}\n No information provided: {} \n Non-existing in the database: {}\n'.format(
                                    col, len(intersections[col]), ','.join(differences_in[col]),
                                    ','.join(differences_out[col]))
                    else:
                        message = 'ERROR: Format of the Clinical Data file is not correct. Check template in the documentation. Check columns: subject external_id, biological_sample external_id and analytical_sample external_id'
                        builder_utils.remove_directory(directory)

                        return message, style, style, table
            try:
                for dataset in datasets:
                    if dataset != "experimental_design":
                        source = os.path.join(temporaryDirectory, dataset)
                        destination = os.path.join(destDir, dataset)
                        builder_utils.copytree(source, destination)
                        datasetPath = os.path.join(
                            os.path.join(ckg_config['imports_experiments_directory'], project_id), dataset)
                        eh.generate_dataset_imports(project_id, dataset, datasetPath)

                loader.partialUpdate(imports=['experiment'], specific=[project_id])
                filename = os.path.join(ckg_config['tmp_directory'], 'Uploaded_files_' + project_id)
                utils.compress_directory(filename, temporaryDirectory, compression_format='zip')
                style.update({'display': 'inline-block'})
                message = 'Files successfully uploaded.'
                table = dataUpload.get_project_information(driver, project_id)
                if table is None:
                    message = 'Error: No data was uploaded for project: {}. Review your experimental design and data files.'.format(
                        project_id)
            except Exception as err:
                style.update({'display': 'none'})
                message = str(err)
        else:
            style.update({'display': 'none'})
            message = "ERROR: Database is offline. Contact your administrator or start the database."

    return message, style, style, table


@dash.callback(Output('upload-result', 'style'),
               [Input('upload-result', 'children')])
def change_style_data_upload(upload_result):
    if upload_result is None:
        return {'fontSize': '20px', 'marginLeft': '70%', 'color': 'black'}
    else:
        if 'ERROR' in upload_result:
            return {'fontSize': '20px', 'marginLeft': '70%', 'color': 'red'}
        if 'WARNING' in upload_result:
            return {'fontSize': '20px', 'marginLeft': '70%', 'color': 'orange'}
        else:
            return {'fontSize': '20px', 'marginLeft': '70%', 'color': 'black'}


@dash.callback(Output('data_download_link', 'href'),
               [Input('data_download_link', 'n_clicks'),
                Input('project_id', 'value')])
def generate_upload_zip(n_clicks, project_id):
    session_cookie = flask.request.cookies.get('custom-auth-session')
    return '/tmp/{}_{}'.format(session_cookie + "upload", project_id)
