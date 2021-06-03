import warnings
import os
import shutil
import subprocess
import re
import pandas as pd
import numpy as np
import time
from datetime import datetime
from uuid import uuid4
import base64
import flask
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from ckg import ckg_utils
import ckg.report_manager.user as user
from ckg.report_manager.app import app, server as application
from ckg.report_manager.apps import initialApp, adminApp, projectCreationApp, dataUploadApp, dataUpload, projectApp, importsApp, homepageApp, loginApp, projectCreation
from ckg.graphdb_builder import builder_utils
from ckg.graphdb_builder.builder import loader, builder
from ckg.graphdb_builder.experiments import experiments_controller as eh
from ckg.report_manager import utils
from ckg.report_manager.worker import create_new_project, create_new_identifiers, run_minimal_update_task, run_full_update_task
from ckg.graphdb_connector import connector

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    ckg_config = ckg_utils.read_ckg_config()
    log_config = ckg_config['report_manager_log']
    logger = builder_utils.setup_logging(log_config, key="index page")
    config = builder_utils.setup_config('builder')
    separator = config["separator"]
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))

app.layout = dcc.Loading(children=[html.Div([dcc.Location(id='url', refresh=False),
                                             html.Div(id='page-content',
                                                      style={'padding-top': 10},
                                                      className='container-fluid')])],
                         style={'text-align': 'center',
                                'top': '50%',
                                'left': '50%',
                                'height': '250px'},
                         type='cube', color='#2b8cbe')


@app.callback([Output('page-content', 'children'),
               Output('logout_form', 'style'),
               Output('error_msg', 'style')],
              [Input('url', 'href')])
def display_page(pathname):
    session_cookie = flask.request.cookies.get('custom-auth-session')
    logged_in = session_cookie is not None
    if not logged_in:
        if pathname is not None and 'error' in pathname:
            error = {'display': 'block'}
        else:
            error = {'display': 'none'}
        login_form = loginApp.LoginApp("Login", "", "", layout=[], logo=None, footer=None)
        return (login_form.layout, {'display': 'none'}, error)
    elif pathname is not None:
        if '/apps/initial' in pathname:
            return (initialApp.layout, {'display': 'block',
                                        'position': 'absolute',
                                        'right': '50px'}, {'display': 'none'})
        elif '/apps/login' in pathname:
            if logged_in:
                stats_db = homepageApp.HomePageApp("CKG homepage", "Database Stats", "", layout=[], logo=None, footer=None)
                return (stats_db.layout, {'display': 'block',
                                          'position': 'absolute',
                                          'right': '50px'}, {'display': 'none'})
        elif '/apps/admin' in pathname:
            layout = []
            if 'admin?' in pathname:
                if 'new_user' in pathname:
                    username = pathname.split('=')[1]
                    if 'error' in pathname:
                        layout.append(html.Div(children=[html.H3("– Error creating new user: {} – ".format(username.replace('%20', ' ')))], className='error_panel'))
                    else:
                        layout.append(html.Div(children=[html.H3("– New user successfully created: {} –".format(username))], className='info_panel'))
                elif 'running' in pathname:
                    running_type = pathname.split('=')[1]
                    layout.append(html.Div(children=[html.H3("– The {} update is running. This will take a while, check the logs: graphdb_builder.log for more information –".format(running_type))], className='info_panel'))
            admin_page = adminApp.AdminApp("CKG Admin Dashboard", "Admin Dashboard", "", layout=layout, logo=None, footer=None)
            return (admin_page.layout, {'display': 'block',
                                        'position': 'absolute',
                                        'right': '50px'}, {'display': 'none'})
        elif '/apps/projectCreationApp' in pathname:
            projectCreation_form = projectCreationApp.ProjectCreationApp("Project Creation", "", "", layout=[], logo=None, footer=None)
            return (projectCreation_form.layout, {'display': 'block', 'position': 'absolute', 'right': '50px'}, {'display': 'none'})
        elif '/apps/dataUploadApp' in pathname:
            dataUpload_form = dataUploadApp.DataUploadApp("Data Upload", "", "", layout=[], logo=None, footer=None)
            return (dataUpload_form.layout, {'display': 'block', 'position': 'absolute', 'right': '50px'}, {'display': 'none'})
        elif '/apps/project?' in pathname:
            project_id, force, session_id = get_project_params_from_url(pathname)
            if session_id is None:
                session_id = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
            if project_id is None:
                return (initialApp.layout, {'display': 'block',
                                            'position': 'absolute',
                                            'right': '50px'}, {'display': 'none'})
            else:
                project = projectApp.ProjectApp(session_id, project_id, project_id, "", "", layout=[], logo=None, footer=None, force=force)
                return (project.layout, {'display': 'block',
                                         'position': 'absolute',
                                         'right': '50px'}, {'display': 'none'})
        elif '/apps/imports' in pathname:
            imports = importsApp.ImportsApp("CKG imports monitoring", "Statistics", "", layout=[], logo=None, footer=None)
            return (imports.layout, {'display': 'block',
                                     'position': 'absolute',
                                     'right': '50px'}, {'display': 'none'})
        elif '/apps/homepage' in pathname or pathname.count('/') <= 3:
            stats_db = homepageApp.HomePageApp("CKG homepage", "Database Stats", "", layout=[], logo=None, footer=None)
            return (stats_db.layout, {'display': 'block',
                                      'position': 'absolute',
                                      'right': '50px'}, {'display': 'none'})
        else:
            return ('404', {'display': 'block', 'position': 'absolute', 'right': '50px'}, {'display': 'none'})

    return (None, None, {'display': 'none'})


def get_project_params_from_url(pathname):
    force = False
    project_id = None
    session_id = None
    regex_id = r"project_id=(\w+)"
    regex_force = r"force=(\d)"
    regex_session = r"session=(.+)"
    match_id = re.search(regex_id, pathname)
    if match_id:
        project_id = match_id.group(1)
    match_force = re.search(regex_force, pathname)
    if match_force:
        force = bool(int(match_force.group(1)))
    match_session = re.search(regex_session, pathname)
    if match_session:
        session_id = match_session.group(1)

    return project_id, force, session_id



@app.callback([Output('upload-config', 'style'),
               Output('output-data-upload', 'children'),
               Output('upload-config', 'filename')],
              [Input('upload-config', 'contents'),
               Input('my-dropdown', 'value')],
              [State('upload-config', 'filename')])
def update_output(contents, value, fname):
    display = {'display': 'none'}
    uploaded = None
    if value is not None:
        page_id, dataset = value.split('/')
        if not os.path.exists(ckg_config['tmp_directory']):
            os.makedirs(ckg_config['tmp_directory'])
        directory = os.path.join(ckg_config['tmp_directory'], page_id)
        if dataset != "defaults":
            display = {'width': '50%',
                       'height': '60px',
                       'lineHeight': '60px',
                       'borderWidth': '2px',
                       'borderStyle': 'dashed',
                       'borderRadius': '15px',
                       'textAlign': 'center',
                       'margin-bottom': '20px',
                       'display': 'block'}
            if not os.path.exists(directory):
                os.makedirs(directory)

            if fname is None:
                contents = None
            if contents is not None:
                with open(os.path.join(directory, dataset+'.yml'), 'wb') as out:
                    content_type, content_string = contents.split(',')
                    decoded = base64.b64decode(content_string)
                    out.write(decoded)
                uploaded = dcc.Markdown("**{} configuration uploaded: {}** &#x2705;".format(dataset.title(),fname))
                fname = None
                contents = None
            else:
                uploaded = None
        elif dataset == 'reset':
            display = {'display': 'none'}
            if os.path.exists(directory):
                shutil.rmtree(directory)
    return display, uploaded, fname


@app.callback(Output('db-creation-date', 'children'),
              [Input('db_stats_df', 'data')])
def update_db_date(df):
    db_date = "Unknown"
    if 'kernel_monitor' in df:
        kernel = pd.read_json(df['kernel_monitor'], orient='records')
        db_date = kernel['storeCreationDate'][0]

    return html.H3('Store Creation date: {}'.format(db_date))


@app.callback([Output("db_indicator_14", "children"),
               Output("db_indicator_1", "children"),
               Output("db_indicator_3", "children"),
               Output("db_indicator_2", "children"),
               Output("db_indicator_4", "children"),
               Output("db_indicator_5", "children"),
               Output("db_indicator_6", "children"),
               Output("db_indicator_7", "children"),
               Output("db_indicator_8", "children"),
               Output("db_indicator_9", "children"),
               Output("db_indicator_10", "children"),
               Output("db_indicator_11", "children"),
               Output("db_indicator_12", "children"),
               Output("db_indicator_13", "children"),
               ],
              [Input("db_stats_df", "data")])
def number_panel_update(df):
    updates = []
    if 'projects' in df:
        projects = pd.read_json(df['projects'], orient='records')
        if not projects.empty and 'Projects' in projects:
            projects = projects['Projects'][0]
        updates.append(projects)
        if 'meta_stats' in df:
            meta_stats = pd.read_json(df['meta_stats'], orient='records')
            if not meta_stats.empty:
                if 'nodeCount' in meta_stats:
                    ent = meta_stats['nodeCount'][0]
                else:
                    ent = '0'
                updates.append(ent)
                if 'relCount' in meta_stats:
                    rel = meta_stats['relCount'][0]
                else:
                    rel = '0'
                updates.append(rel)
                if 'labelCount' in meta_stats:
                    labels = meta_stats['labelCount'][0]
                else:
                    labels = '0'
                updates.append(labels)
                if 'relTypeCount' in meta_stats:
                    types = meta_stats['relTypeCount'][0]
                else:
                    types = '0'
                updates.append(types)
                if 'propertyKeyCount' in meta_stats:
                    prop = meta_stats['propertyKeyCount'][0]
                else:
                    prop = '0'
                updates.append(prop)

    if 'store_size' in df:
        store_size = pd.read_json(df['store_size'], orient='records')
        if not store_size.empty and 'size' in store_size:
            ent_store = store_size['size'][2]
            rel_store = store_size['size'][4]
            prop_store = store_size['size'][3]
            string_store = store_size['size'][5]
            array_store = store_size['size'][0]
            log_store = store_size['size'][1]
        else:
            ent_store = '0 MB'
            rel_store = '0 MB'
            prop_store = '0 MB'
            string_store = '0 MB'
            array_store = '0 MB'
            log_store = '0 MB'

        updates.extend([ent_store, rel_store, prop_store, string_store, array_store, log_store])

    if 'transactions' in df:
        transactions = pd.read_json(df['transactions'], orient='records')
        if not transactions.empty and 'name' in transactions:
            t_open = transactions.loc[transactions['name'] == 'NumberOfOpenedTransactions', 'value'].iloc[0]
            t_comm = transactions.loc[transactions['name'] == 'NumberOfCommittedTransactions', 'value'].iloc[0]
        else:
            t_open = '0'
            t_comm = '0'

        updates.extend([t_open, t_comm])

    return [dcc.Markdown("**{}**".format(i)) for i in updates]


@app.callback(Output("project_url", "children"),
              [Input("project_option", "value")])
def update_project_url(value):
    if value is not None and len(value) > 1:
        return html.A(value[0].title(),
                      href='/apps/project?project_id={}&force=0'.format(value[1]),
                      target='',
                      n_clicks=0,
                      className="button_link")
    else:
        return ''


@app.server.route('/apps/login', methods=['POST', 'GET'])
def route_login():
    data = flask.request.form
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        flask.abort(401)
    elif not user.User(username).verify_password(password):
        return flask.redirect('/login_error')
    else:
        rep = flask.redirect('/')
        rep.set_cookie('custom-auth-session', username+'_'+datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()))
        return rep


@app.server.route('/apps/logout', methods=['POST'])
def route_logout():
    # Redirect back to the index and remove the session cookie.
    rep = flask.redirect('/')
    rep.set_cookie('custom-auth-session', '', expires=0)

    return rep


@app.server.route('/create_user', methods=['POST', 'GET'])
def route_create_user():
    data = flask.request.form
    name = data.get('name')
    surname = data.get('surname')
    affiliation = data.get('affiliation')
    acronym = data.get('acronym')
    email = data.get('email')
    alt_email = data.get('alt_email')
    phone = data.get('phone')
    uname = name[0] + surname
    username = uname

    registered = 'error_exists'
    iter = 1
    while registered == 'error_exists':
        u = user.User(username=username.lower(), name=name, surname=surname, affiliation=affiliation, acronym=acronym, phone=phone, email=email, alternative_email=alt_email)
        registered = u.register()
        if registered is None:
            rep = flask.redirect('/apps/admin?error_new_user={}'.format('Failed Database'))
        elif registered == 'error_exists':
            username = uname + str(iter)
            iter += 1
        elif registered == 'error_email':
            rep = flask.redirect('/apps/admin?error_new_user={}'.format('Email already registered'))
        elif registered == 'error_database':
            rep = flask.redirect('/apps/admin?error_new_user={}'.format('User could not be saved in the database'))
        else:
            rep = flask.redirect('/apps/admin?new_user={}'.format(username))

    return rep


@app.server.route('/update_minimal', methods=['POST', 'GET'])
def route_minimal_update():
    session_cookie = flask.request.cookies.get('custom-auth-session')
    username = session_cookie.split('_')[0]
    internal_id = datetime.now().strftime('%Y%m-%d%H-%M%S-')
    result = run_minimal_update_task.apply_async(args=[username], task_id='run_minimal_'+session_cookie+internal_id, queue='update')

    rep = flask.redirect('/apps/admin?running=minimal')

    return rep


@app.server.route('/update_full', methods=['POST', 'GET'])
def route_full_update():
    session_cookie = flask.request.cookies.get('custom-auth-session')
    data = flask.request.form
    download = data.get('dwn-radio') == 'true'
    username = session_cookie.split('_')[0]
    internal_id = datetime.now().strftime('%Y%m-%d%H-%M%S-')
    result = run_full_update_task.apply_async(args=[username, download], task_id='run_full_'+session_cookie+internal_id, queue='update')
    
    rep = flask.redirect('/apps/admin/running=full')

    return rep


@app.callback(Output('download-zip', 'href'),
              [Input('download-zip', 'n_clicks')],
              [State('url', 'href')])
def generate_report_url(n_clicks, pathname):
    project_id, force, session_id = get_project_params_from_url(pathname)
    return '/downloads/{}'.format(project_id)


@application.route('/downloads/<value>')
def route_report_url(value):
    uri = os.path.join(ckg_config['downloads_directory'], value + '.zip')
    return flask.send_file(uri, attachment_filename=value + '.zip', as_attachment=True, cache_timeout=-1)

@application.route('/example_files')
def route_example_files_url():
    uri = os.path.join(ckg_config['data_directory'], 'example_files.zip')
    return flask.send_file(uri, attachment_filename='example_files.zip', as_attachment=True, cache_timeout=-1)

###Callback regenerate project
@app.callback(Output('regenerate', 'href'),
              [Input('regenerate', 'n_clicks'),
               Input('regenerate', 'title')],
              [State('url', 'href')])
def regenerate_report(n_clicks, title, pathname):
    basic_path = '/'.join(pathname.split('/')[0:3])
    project_id, force, session_id = get_project_params_from_url(pathname)
    return basic_path+'/apps/project?project_id={}&force=1&session={}'.format(project_id, title)


###Callbacks for project creation app
def image_formatter(im):
    data_im = base64.b64encode(im).decode('ascii')
    return f'<img src="data:image/jpeg;base64,{data_im}">'


@app.callback([Output('project-creation', 'children'),
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
               #State('number_subjects', 'value'),
               State('project description', 'value'),
               State('date-picker-start', 'date'),
               State('date-picker-end', 'date')])
def create_project(n_clicks, name, acronym, responsible, participant, datatype, timepoints, related_to, disease, tissue, intervention, description, start_date, end_date):
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
                        response = 'The intervention(s) "{}" specified does(do) not exist.'.format(', '.join([k for k,n in exist.items() if n==True]))
                        return response, None, {'display': 'none'}, {'display': 'none'}

            if any(not arguments[n] for n, i in enumerate(arguments)):
                response = "Insufficient information to create project. Fill in all fields with '*'."
                return response, None, {'display': 'none'}, {'display': 'none'}

            # Get project data from filled-in fields
            projectData = pd.DataFrame([name, acronym, description, related_to, datatype, timepoints, disease, tissue, intervention, responsible, participant, start_date, end_date]).T
            projectData.columns = ['name', 'acronym', 'description', 'related_to', 'datatypes', 'timepoints', 'disease', 'tissue', 'intervention', 'responsible', 'participant', 'start_date', 'end_date']
            projectData['status'] = ''

            projectData.fillna(value=pd.np.nan, inplace=True)
            projectData.replace('', np.nan, inplace=True)

            # Generate project internal identifier bsed on timestamp
            # Excel file is saved in folder with internal id name
            epoch = time.time()
            internal_id = "%s%d" % ("CP", epoch)
            projectData.insert(loc=0, column='internal_id', value=internal_id)
            result = create_new_project.apply_async(args=[internal_id, projectData.to_json(), separator], task_id='project_creation_'+session_cookie+internal_id, queue='creation')
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
        
        return response, '- '+external_id, {'display': 'inline-block'}, {'display': 'block'}
    else:
        return None, None, {'display': 'none'}, {'display': 'none'}


@app.callback(Output('project-creation', 'style'),
              [Input('project-creation', 'children')])
def change_style(style):
    if style is not None and 'successfully' in style:
        return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'black'}
    else:
        return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'red'}


@app.callback(Output('download_link', 'href'),
              [Input('update_project_id', 'children')])
def update_download_link(project):
    if project is not None and project != '':
        return '/apps/templates{}'.format('Design_and_Clinical_templates')
    else:
        return ''

@application.route('/apps/templates<value>')
def serve_static(value):
    cwd = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(cwd,'apps/templates/')
    filename = os.path.join(directory, value)
    url = filename+'.zip'
    if not os.path.isfile(url):
        utils.compress_directory(filename, os.path.join(directory, 'files'), compression_format='zip')

    return flask.send_file(url, attachment_filename = value+'.zip', as_attachment = True, cache_timeout=-1)


###Callbacks for data upload app
@app.callback([Output('existing-project', 'children'),
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


@app.callback(Output('proteomics-tool', 'style'),
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


@app.callback([Output('proteomics-file', 'style'),
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


@app.callback([Output('uploaded-files', 'children'),
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
        if dataset in ['proteomics', 'interactomics', 'phosphoproteomics'] and prot_tool != '' and (prot_file != '' or prot_tool == 'mzTab'):
            selected_file = prot_tool.lower() + "-" + prot_file.lower()
            if selected_file in config['file_proteomics']:
                filename = config['file_proteomics'][selected_file]
            else:
                if prot_tool == 'mzTab':
                    filename = dataset+'_'+prot_tool.lower()+'.mztab'
                else:
                    filename = dataset+'_'+prot_tool.lower()+'_'+prot_file.replace(' ', '').lower()+'.'+uploaded_file.split('.')[-1]
            directory = os.path.join(directory, prot_tool.lower())
            if os.path.exists(directory):
                if os.path.exists(os.path.join(directory, filename)):
                    os.remove(os.path.join(directory, filename))
            builder_utils.checkDirectory(directory)
        elif dataset == 'experimental_design':
            filename = config['file_design'].split('_')[0]+'_'+projectid+'.'+uploaded_file.split('.')[-1]
        elif dataset == 'clinical':
            filename = config['file_clinical'].split('_')[0]+'_'+projectid+'.'+uploaded_file.split('.')[-1]

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


@app.callback([Output('upload-result', 'children'),
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
        temporaryDirectory = os.path.join(ckg_config['tmp_directory'], session_cookie+"upload")
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
                        result = create_new_identifiers.apply_async(args=[project_id, designData.to_json(), directory, experimental_filename], task_id='data_upload_'+session_cookie+datetime.now().strftime('%Y%m-%d%H-%M%S-'), queue='creation')
                        result_output = result.wait(timeout=None, propagate=True, interval=0.2)
                        res_n = pd.DataFrame.from_dict(result_output['res_n'])
                        builder_utils.copytree(directory, destination)
                    else:
                        message = 'ERROR: The Experimental design file provided ({}) is missing some of the required fields: {}'.format(experimental_filename, ','.join(['subject external_id','biological_sample external_id','analytical_sample external_id']))
                        builder_utils.remove_directory(directory)

                        return message, style, style, table

            if 'clinical' in datasets:
                dataset = 'clinical'
                directory = os.path.join(temporaryDirectory, dataset)
                clinical_files = os.listdir(directory)
                regex = r"{}.+".format(config['file_clinical'].replace('PROJECTID', project_id) )
                r = re.compile(regex)
                clinical_filename = list(filter(r.match, clinical_files))
                if len(clinical_filename) > 0:
                    clinical_filename = clinical_filename.pop()
                    data = builder_utils.readDataset(os.path.join(directory, clinical_filename))
                    data.columns = [c.lower() for c in data.columns]
                    external_ids = {}
                    if 'subject external_id' in data and 'biological_sample external_id' in data:
                        external_ids['subjects'] = data['subject external_id'].astype(str).unique().tolist()
                        external_ids['biological_samples'] = data['biological_sample external_id'].astype(str).unique().tolist()
                        dataUpload.create_mapping_cols_clinical(driver, data, directory, clinical_filename, separator=separator)
                        if 0 in res_n.values:
                            samples = ', '.join([k for (k,v) in res_n if v == 0])
                            message = 'ERROR: No {} for project {} in the database. Please upload first the experimental design (ExperimentalDesign_{}.xlsx)'.format(samples, project_id, project_id)
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
                                message += 'WARNING: Some {} identifiers were not matched:\n Matching: {}\n No information provided: {} \n Non-existing in the database: {}\n'.format(col, len(intersections[col]), ','.join(differences_in[col]), ','.join(differences_out[col]))
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
                        datasetPath = os.path.join(os.path.join(ckg_config['imports_experiments_directory'], project_id), dataset)
                        eh.generate_dataset_imports(project_id, dataset, datasetPath)

                loader.partialUpdate(imports=['experiment'], specific=[project_id])
                filename = os.path.join(ckg_config['tmp_directory'], 'Uploaded_files_'+project_id)
                utils.compress_directory(filename, temporaryDirectory, compression_format='zip')
                style.update({'display':'inline-block'})
                message = 'Files successfully uploaded.'
                table = dataUpload.get_project_information(driver, project_id)
                if table is None:
                    message = 'Error: No data was uploaded for project: {}. Review your experimental design and data files.'.format(project_id)
            except Exception as err:
                style.update({'display':'none'})
                message = str(err)
        else:
            style.update({'display':'none'})
            message = "ERROR: Database is offline. Contact your administrator or start the database."

    return message, style, style, table

@app.callback(Output('upload-result', 'style'),
              [Input('upload-result', 'children')])
def change_style_data_upload(upload_result):
    if upload_result is None:
        return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'black'}
    else:
        if 'ERROR' in upload_result:
            return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'red'}
        if 'WARNING' in upload_result:
            return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'orange'}
        else:
            return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'black'}

@app.callback(Output('data_download_link', 'href'),
             [Input('data_download_link', 'n_clicks'),
              Input('project_id', 'value')])
def generate_upload_zip(n_clicks, project_id):
    session_cookie = flask.request.cookies.get('custom-auth-session')
    return '/tmp/{}_{}'.format(session_cookie+"upload", project_id)

@application.route('/tmp/<value>')
def route_upload_url(value):
    page_id, project_id = value.split('_')
    directory = ckg_config['tmp_directory']
    filename = os.path.join(directory, 'Uploaded_files_'+project_id)
    url = filename+'.zip'

    return flask.send_file(url, attachment_filename = filename.split('/')[-1]+'.zip', as_attachment = True, cache_timeout=-1)

def main():
    print("IN MAIN")
    celery_working_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(celery_working_dir)
    queues = [('creation', 1, 'INFO'), ('compute', 3, 'INFO'), ('update', 1, 'INFO')]
    for queue, processes, log_level in queues:
        celery_cmdline = 'celery -A ckg.report_manager.worker worker --loglevel={} --concurrency={} -E -Q {}'.format(log_level, processes, queue).split(" ")
        print("Ready to call {} ".format(celery_cmdline))
        subprocess.Popen(celery_cmdline)
        print("Done callling {} ".format(celery_cmdline))
    
    application.run(debug=False, host='0.0.0.0')  


if __name__ == '__main__':
    main()
