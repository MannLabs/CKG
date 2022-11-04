import base64
import os
import shutil
from datetime import datetime
from uuid import uuid4

import dash
from dash import html, dcc, Output, Input, State

from ckg import ckg_utils
from ckg.report_manager import project
from ckg.report_manager.worker import generate_project_report

title = "Project details"
subtitle = "You are successfully authorized"
description = ""

dash.register_page(__name__, path='/apps/project', title=f"{title} - {subtitle}", description=description)


def layout(project_id="P0000001", force=0):
    print(project_id)
    print(force)

    session_id = project_id + datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

    # inital_layout = [html.H1(children=title),
    #                 html.H2(children=subtitle),
    #                 html.Div(children=description)]
    project_layout = build_page(project_id, force, session_id)
    return project_layout


def build_page(project_id, force, session_id):
    """
    Builds project and generates the report.
    For each data type in the report (e.g. 'proteomics', 'clinical'), \
    creates a designated tab.
    A button to download the entire project and report is added.
    """
    print("Build page")
    config_files = {}
    tmp_dir = ckg_utils.read_ckg_config(key='tmp_directory')
    if os.path.exists(tmp_dir):
        directory = os.path.join(tmp_dir, project_id)
        if os.path.exists(directory):
            config_files = {f.split('.')[0]: os.path.join(directory, f) for f in os.listdir(directory) if
                            os.path.isfile(os.path.join(directory, f))}

    print("Finished zip")

    result = generate_project_report.apply_async(args=[project_id, config_files, force],
                                                 task_id='generate_report' + session_id, queue='compute')
    result_output = result.get()
    print("Project result")
    print(result_output == None)
    print(result_output)

    p = project.Project(project_id, datasets={}, knowledge=None, report={}, configuration_files=config_files)
    p.build_project(False)

    print("Get project")

    # TODO: fix this
    if p.name is not None:
        title = "Project: {}".format(p.name)
    else:
        title = ''
    plots = p.show_report("app")
    print("Plots:")
    print(plots)
    p = None
    tabs = []
    buttons = build_header(project_id, session_id)
    print("build header")

    layout = []
    layout.append(buttons)
    for data_type in plots:
        if len(plots[data_type]) >= 1:
            tab_content = [html.Div(plots[data_type])]
            tab = dcc.Tab(tab_content, label=data_type)
            tabs.append(tab)
    lc = dcc.Tabs(tabs)
    layout.append(lc)
    print(layout)
    return layout


def build_header(project_id, session_id):
    buttons = html.Div([html.Div([html.A('Download Project Report',
                                         id='download-zip',
                                         href=f"/downloads/{project_id}",
                                         target="_blank",
                                         n_clicks=0,
                                         className="button_link"
                                         )]),
                        html.Div([html.A("Regenerate Project Report",
                                         id='regenerate',
                                         title=project_id,
                                         # TODO: {basic_path}?
                                         # basic_path = '/'.join(pathname.split('/')[0:3])
                                         href=f"/apps/project?project_id={project_id}&force=1&session={session_id}",
                                         target='',
                                         n_clicks=0,
                                         className="button_link")]),
                        html.Div([html.H3("Change Analysis' Configuration: "),
                                  dcc.Dropdown(
                                      id='my-dropdown',
                                      options=[
                                          {'label': '', 'value': project_id + '/defaults'},
                                          {'label': 'Proteomics configuration', 'value': project_id + '/proteomics'},
                                          {'label': 'Interactomics configuration',
                                           'value': project_id + '/interactomics'},
                                          {'label': 'Phosphoproteomics configuration',
                                           'value': project_id + '/phosphoproteomics'},
                                          {'label': 'Clinical data configuration', 'value': project_id + '/clinical'},
                                          {'label': 'Multiomics configuration', 'value': project_id + '/multiomics'},
                                          {'label': 'Reset to defaults', 'value': project_id + '/reset'}],
                                      value=project_id + '/defaults',
                                      clearable=False,
                                      style={'width': '50%', 'margin-bottom': '10px'}),
                                  dcc.Upload(id='upload-config',
                                             children=html.Div(['Drag and Drop or ',
                                                                html.A('Select Files')]),
                                             max_size=-1,
                                             multiple=False),
                                  html.Div(id='output-data-upload')])
                        ])

    return buttons


@dash.callback([Output('upload-config', 'style'),
                Output('output-data-upload', 'children'),
                Output('upload-config', 'filename')],
               [Input('upload-config', 'contents'),
                Input('my-dropdown', 'value')],
               [State('upload-config', 'filename')])
def update_output(contents, value, fname):
    ckg_config = ckg_utils.read_ckg_config()
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
                with open(os.path.join(directory, dataset + '.yml'), 'wb') as out:
                    content_type, content_string = contents.split(',')
                    decoded = base64.b64decode(content_string)
                    out.write(decoded)
                uploaded = dcc.Markdown("**{} configuration uploaded: {}** &#x2705;".format(dataset.title(), fname))
                fname = None
                contents = None
            else:
                uploaded = None
        elif dataset == 'reset':
            display = {'display': 'none'}
            if os.path.exists(directory):
                shutil.rmtree(directory)
    return display, uploaded, fname
