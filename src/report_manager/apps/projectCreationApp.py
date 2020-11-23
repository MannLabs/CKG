from apps import basicApp
import dash_core_components as dcc
import dash_html_components as html
from graphdb_connector import connector


driver = connector.getGraphDatabaseConnectionConfiguration()
DataTypes = ['clinical', 'proteomics',
             'interactomics', 'phosphoproteomics',
             'longitudinal_proteomics', 'longitudinal_clinical']


class ProjectCreationApp(basicApp.BasicApp):
    """
    Defines what the project creation App is in the report_manager.
    Includes multiple fill in components to gather project information and metadata.
    """
    def __init__(self, title, subtitle, description, layout=[], logo=None, footer=None):
        self.pageType = "projectCreationPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        """
        Builds page with the basic layout from *basicApp.py* and adds relevant Dash components for project creation.
        """
        self.add_basic_layout()
        try:
            Users = [(u['name']) for u in driver.nodes.match("User")]
            Tissues = [(t['name']) for t in driver.nodes.match("Tissue")]
            Diseases = [(d['name']) for d in driver.nodes.match("Disease")]
            layout = [html.Div([
                    html.Div([html.H4('Project information', style={'width': '15.5%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.H4('', id='update_project_id', style={'width': '15%', 'verticalAlign': 'top', 'display': 'none'}),
                              html.Br(),
                              html.Div(children=[html.Label('Project name:*', style={'marginTop': 15}),
                                                 dcc.Input(id='project name', placeholder='Insert name...', type='text', style={'width': '100%', 'height': '35px'})],
                                                 style={'width': '100%'}),
                              html.Br(),
                              html.Div(children=[html.Label('Project Acronym:', style={'marginTop': 15}),
                                                 dcc.Input(id='project acronym', placeholder='Insert name...', type='text', style={'width': '100%', 'height': '35px'})],
                                                 style={'width': '100%'}),
                              html.Br(),
                              html.Div(children=[html.Label('Project Responsible:*', style={'marginTop': 15})],
                                                 style={'width': '49%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Div(children=[html.Label('Project Participants:*', style={'marginTop': 15})],
                                                 style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='responsible-picker', options=[{'label': i, 'value': i} for i in Users], value=[], multi=True, searchable=True, style={'width': '100%'})],
                                                 style={'width': '49%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='participant-picker', options=[{'label': i, 'value': i} for i in Users], value=[], multi=True, searchable=True, style={'width': '100%'})],
                                                 style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Br(),
                              html.Br(),
                              html.Div(children=[html.Label('Project Data Types:*', style={'marginTop': 10})],
                                                 style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Div(children=[html.Label('Project Disease:*', style={'marginTop': 10})],
                                                 style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='data-types-picker', options=[{'label': i, 'value': i} for i in DataTypes], value=[], multi=True, searchable=True, style={'width': '100%'})],
                                                 style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='disease-picker', options=[{'label': i, 'value': i} for i in Diseases], value=[], multi=True, searchable=True, style={'width': '100%'})],
                                                 style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Br(),
                              html.Br(),
                              html.Div(children=[html.Label('Project Tissue:*', style={'marginTop': 10})],
                                                 style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Div(children=[html.Label('Project Intervention:', style={'marginTop': 10})],
                                                 style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Div(children=[dcc.Dropdown(id='tissue-picker', options=[{'label': i, 'value': i} for i in Tissues], value=[], multi=True, searchable=True, style={'width': '100%'})],
                                                 style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Div(children=[dcc.Input(id='intervention-picker', placeholder='E.g. SNOMED identifier|SNOMED identifier|...', type='text', style={'width': '100%', 'height': '54px'})],
                                                 style={'width': '49%', 'marginLeft': '2%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Br(),
                              html.Br(),
                              html.Div(children=[html.Label('Timepoints:', style={'marginTop': 15}),
                                                 dcc.Input(id='number_timepoints', placeholder='E.g. 2 months|15 days|24 hours...', type='text', style={'width': '100%', 'height': '35px'})],
                                                 style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Br(),
                              html.Br(),
                              html.Div(children=[html.Label('Follows up project:', style={'marginTop': 15}),
                                                 dcc.Input(id='related_to', placeholder='Use the Project Identifier (P000000X)', type='text', style={'width': '100%', 'height': '35px'})],
                                                 style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top', 'display': 'inline-block'}),
                              html.Br(),
                              html.Br(),
                              html.Div(children=[html.Label('Project Description:', style={'marginTop': 15}),
                                                 dcc.Textarea(id='project description', placeholder='Enter description...', style={'width': '100%', 'height': '100px'})]),
                              html.Br(),
                              html.Div(children=[html.Label('Starting Date:', style={'marginTop': 10}),
                                                 dcc.DatePickerSingle(id='date-picker-start', placeholder='Select date...', clearable=True)],
                                                 style={'width': '30%', 'verticalAlign': 'top', 'marginTop': 10, 'display': 'inline-block'}),
                              html.Div(children=[html.Label('Ending Date:', style={'marginTop': 10}),
                                                 dcc.DatePickerSingle(id='date-picker-end', placeholder='Select date...', clearable=True)],
                                                 style={'width': '30%', 'verticalAlign': 'top', 'marginTop': 10, 'display': 'inline-block'}),
                              html.Div(children=html.Button('Create Project', id='project_button', n_clicks=0, className="button_link",
                                                            style={'position': 'absolute', 'right': '0px'})),
                              html.Br(),
                              html.Div(children=[html.A(children=html.Button('Download Clinical Data template', id='download_button', n_clicks=0,
                                                                              style={'fontSize': '16px', 'display': 'block'}),
                                                 id='download_link', href='', n_clicks=0)], style={'width': '100%', 'padding-left': '87%', 'padding-right': '0%'}),
                              html.Br(),
                              html.Div(children=[html.H1(id='project-creation')]),
                              html.Br()]),
                    html.Hr()])]
        except Exception as e:
            layout = [html.Div(children=html.H1("Database is offline", className='error_msg'))]

        self.extend_layout(layout)