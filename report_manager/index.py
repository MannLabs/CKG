import io
import os
import re
import pandas as pd
import time
from datetime import datetime
import base64
import qrcode
import barcode
import flask
import urllib.parse
from IPython.display import HTML

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash_network import Network

from app import app
from apps import initialApp, projectApp, importsApp, projectCreationApp, dataUploadApp, IDRetriver
from graphdb_builder import builder_utils
from graphdb_builder.builder import loader
from graphdb_builder.experiments import experiments_controller as eh
import ckg_utils
import config.ckg_config as ckg_config

import py2neo
import logging
import logging.config
from graphdb_connector import connector

from rq import Queue
from rq.job import Job
from apps.worker import conn


driver = connector.getGraphDatabaseConnectionConfiguration()
# q = Queue(connection=conn)

config = ckg_utils.get_configuration('../graphdb_builder/experiments/experiments_config.yml')



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
        elif pathname.startswith('/apps/dataUpload'):
            projectId = pathname.split('/')[-1]
            dataUpload = dataUploadApp.DataUploadApp(projectId, "Data Upload", "", "", layout = [], logo = None, footer = None)
            return dataUpload.layout
        elif pathname.startswith('/apps/project'):
            projectId = pathname.split('/')[-1]
            project = projectApp.ProjectApp(projectId, projectId, "", "", layout = [], logo = None, footer = None)
            return project.layout
        elif pathname.startswith('/apps/imports'):
            imports = importsApp.ImportsApp("CKG imports monitoring", "Statistics", "", layout = [], logo = None, footer = None)
            return imports.layout
        else:
            return '404'



###Callbacks for project creation app
def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


@app.callback(Output('download_link', 'href'),
             [Input('download_button', 'n_clicks')])
def update_download_link(n_clicks=0):
    relative_filename = os.path.join('apps/templates', 'ClinicalData_template.xlsx')
    absolute_filename = os.path.join(os.getcwd(), relative_filename)
    if n_clicks is not None and n_clicks > 0:
        return '/{}'.format(relative_filename)


@app.server.route('/apps/templates/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'apps/templates'), path)


@app.callback(Output('dum-div', 'children'),
             [Input('responsible', 'value'),
              Input('data-types', 'value'),
              Input('participant', 'value'),
              Input('tissue', 'value'),
              Input('upload-data-type', 'value')])
def update_input(responsible, datatype, participant, tissue, upload_dt):
    return responsible, datatype, participant, tissue, upload_dt


@app.callback(Output('responsible', 'value'),
             [Input('add_responsible', 'n_clicks')],
             [State('responsible-picker','value')])
def update_dropdown(n_clicks, value):
    if n_clicks != None:
        return ', '.join(value)


@app.callback(Output('data-types', 'value'),
             [Input('add_datatype', 'n_clicks')],
             [State('data-types-picker','value')])
def update_dropdown(n_clicks, value):
    if n_clicks != None:
        return ', '.join(value)


@app.callback(Output('participant', 'value'),
             [Input('add_participant', 'n_clicks')],
             [State('participant-picker','value')])
def update_dropdown(n_clicks, value):
    if n_clicks != None:
        return ', '.join(value)


@app.callback(Output('tissue', 'value'),
             [Input('add_tissue', 'n_clicks')],
             [State('tissue-picker','value')])
def update_dropdown(n_clicks, value):
    if n_clicks != None:
        return ', '.join(value)


@app.callback(Output('project-creation', 'children'),
             [Input('project_button', 'n_clicks')],
             [State('project name', 'value'),
              State('project acronym', 'value'),
              State('responsible', 'value'),
              State('data-types', 'value'),
              State('participant', 'value'),
              State('tissue', 'value'),
              State('project description', 'value'),
              State('date-picker-start', 'date'),
              State('date-picker-end', 'date')])
def create_project(n_clicks, name, acronym, responsible, datatype, participant, tissue, description, start_date, end_date):
    if n_clicks != None:
        # Get project data from filled-in fields
        projectData = pd.DataFrame([name, acronym, description, datatype, tissue, responsible, participant, start_date, end_date]).T
        projectData.columns = ['ProjectName', 'ProjectAcronym', 'ProjectDescription', 'ProjectDataTypes', 'ProjectTissue', 'ProjectResponsible', 'ProjectParticipant', 'ProjectStartDate', 'ProjectEndDate']
        projectData['ProjectStatus'] = ''

        # Generate project internal identifier bsed on timestamp
        # Excel file is saved in folder with internal id name
        epoch = time.time()
        internal_id = "%s%d" % ("P", epoch)
        
        projectData.insert(loc=0, column='Project internal_id', value=internal_id)
        
        dataDir = '../../data/experiments/PROJECTID/clinical/'.replace("PROJECTID", internal_id)
        ckg_utils.checkDirectory(dataDir)

        project_csv_string = projectData.to_excel(os.path.join(dataDir, 'ProjectData.xlsx'), index=False, encoding='utf-8')
        #project_csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(project_csv_string)

        ###QUEUE
        # job = q.enqueue_call(func=projectCreationQueue.project_app_importer, args=(internal_id,), result_ttl=5000)
        # print(job.get_id())
        # job2 = q.enqueue_call(func=projectCreationQueue.project_app_loader, args=(driver,internal_id,), result_ttl=5000)
        # print(job2.get_id())

        #Creates project .csv in /imports 
        IDRetriver.project_app_importer(internal_id)

        #Loads project .csv into the database
        IDRetriver.project_app_loader(driver, internal_id)

        return "Project successfully submitted."


@app.callback(Output('project_button', 'disabled'),
             [Input('project_button', 'n_clicks')])
def disable_submit_button(n_clicks):
    if n_clicks > 0:
        return True



###Callbacks for data upload app
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    file = filename.split('.')[-1]
    
    if file == 'txt':
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t', low_memory=False)
    elif file == 'csv':
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), low_memory=False)
    elif file == 'xlsx' or file == 'xls':
        df = pd.read_excel(io.BytesIO(decoded))        
    return df

def export_contents(data, dataDir, filename):
    file = filename.split('.')[-1]
    
    if file == 'txt':
        csv_string = data.to_csv(os.path.join(dataDir, filename), sep='\t', index=False, encoding='utf-8')
    elif file == 'csv':
        csv_string = data.to_csv(os.path.join(dataDir, filename), sep=',', index=False, encoding='utf-8')
    elif file == 'xlsx' or file == 'xls':
        csv_string = data.to_excel(os.path.join(dataDir, filename), index=False, encoding='utf-8')   
    return csv_string

def attribute_internal_ids(data, column, first_id):
    prefix = re.split(r'(^[^\d]+)', first_id)[1]
    id_value = int(re.split(r'(^[^\d]+)', first_id)[-1])
    
    mapping = {}
    for i in data[column].unique():
        new_id = prefix+str(id_value)
        mapping[i] = new_id
        id_value += 1
    
    return mapping


@app.callback([Output('clinical-table', 'data'),
               Output('clinical-table', 'columns')],
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('editing-columns-button', 'n_clicks')],
              [State('clinical-variables-picker', 'value'),
               State('clinical-table', 'columns')])
def update_data(contents, filename, n_clicks, value, existing_columns):
    if contents is not None:
        columns = []
        df = parse_contents(contents, filename)
        if len(df.columns) > 100 and len(df.index) > 1000:
          df = df.iloc[:100,:100]

        data = None
        if df is not None:
            data = df.to_dict('rows')
            for i in df.columns:
                columns.append({'id': i, 'name': i,
                                         'editable_name': False, 'deletable': True})

        if n_clicks is not None and n_clicks > 0:
            for j in value:
                columns.append({'id': j, 'name': j,
                                     'editable_name': False, 'deletable': True})

        empty_cols = [i for i, d in enumerate(columns) if d['id']=='']
        for i in empty_cols:
            del columns[i]

        return data, columns


@app.callback(Output('upload-data-type', 'value'),
             [Input('add_upload_datatype', 'n_clicks')],
             [State('upload-data-type-picker','value')])
def update_dropdown(n_clicks, value):
    if n_clicks != None:
        return value


@app.callback([Output('data_download_link', 'href'),
               Output('data_download_link', 'download')],
              [Input('data_download_button', 'n_clicks')],
              [State('clinical-table', 'columns'),
               State('clinical-table', 'data'),
               State('upload-data-type', 'value')])
def update_table_download_link(n_clicks, columns, rows, data_type):
    if n_clicks != None:
        cols = [d['id'] for d in columns]
        df = pd.DataFrame(rows, columns=cols)
        csv_string = df.to_csv(index=False, encoding='utf-8', sep=';') 
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string, 'downloaded_DATATYPE_DataUpload.csv'.replace('DATATYPE', data_type)


@app.callback(Output('data-upload', 'children'),
             [Input('submit_button', 'n_clicks')],
             [State('clinical-table', 'columns'),
              State('clinical-table', 'data'),
              State('upload-data', 'filename'),
              State('url', 'pathname'),
              State('upload-data-type', 'value')])
def update_table_download_link(n_clicks, columns, rows, filename, path_name, data_type):
    if n_clicks != None:
        # Get Clinical data from Uploaded and updated table
        cols = [d['id'] for d in columns]
        data = pd.DataFrame(rows, columns=cols)

        project_external_id = path_name.split('/')[-1]
        
        #Retrieve identifiers from database
        project_id, subject_id, biosample_id, anasample_id = IDRetriver.retrieve_identifiers_from_database(driver, project_external_id)

        #Add subject, biosample and anasample id columns to data
        data.insert(loc=0, column='subject id', value=data['subject external id'].map(attribute_internal_ids(data, 'subject external id', subject_id).get))
        data.insert(loc=1, column='biological_sample id', value=data['biological_sample external id'].map(attribute_internal_ids(data, 'biological_sample external id', biosample_id).get))
        data.insert(loc=2, column='analytical_sample id', value=data['analytical_sample external id'].map(attribute_internal_ids(data, 'analytical_sample external id', anasample_id).get))

        # # Path to new local folder
        dataDir = '../../data/experiments/PROJECTID/DATATYPE/'.replace('PROJECTID', project_id).replace('DATATYPE', data_type)
        
        # # Check/create folders based on local
        ckg_utils.checkDirectory(dataDir)

        csv_string = export_contents(data, dataDir, filename)
        
        message = 'FILE successfully uploaded.'.replace('FILE', '"'+filename+'"')

        return message



# @app.server.route('/apps/downloads/<path:path>')
# def export_excel_file(path):
#     root_dir = os.getcwd()
#     return flask.send_from_directory(os.path.join(root_dir, 'apps/downloads'), path)


#############

#         localimagefolder = os.path.join(dataDir, 'QRCodes')


#         # Generate QR code per row and save as png
#         images = []
#         for i, row in clinicalData.iterrows():
#             subject, biosample, ansample = row['subject id'], row['biological_sample id'], row['analytical_sample id']

#             filename = project_id+"_"+subject+"_"+biosample+"_"+ansample+".png"

#             qr = qrcode.QRCode(version=1,
#                                error_correction=qrcode.constants.ERROR_CORRECT_L,
#                                box_size=10,
#                                border=4)
#             qr.add_data(project_id+"_"+subject+"_"+biosample+"_"+ansample)
#             qr.make()
#             img = qr.make_image()
#             imagepath = os.path.join(localimagefolder, project_id+"_"+subject+"_"+biosample+"_"+ansample+".png")
#             img.save(imagepath) # Save image
#             images.append(imagepath)

#         with open(os.path.join(localimagefolder, "output.pdf"), "wb") as f:
#             f.write(img2pdf.convert([i for i in images]))

#         # Add png names as new column in dataframe
#         clinicalData['QR code'] = images



if __name__ == '__main__':
    app.run_server(debug=True, port=5000)
