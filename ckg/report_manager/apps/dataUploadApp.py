import dash_core_components as dcc
import dash_html_components as html
from ckg.report_manager.apps import basicApp
from ckg import ckg_utils


DataTypes = ['experimental_design', 'clinical', 'proteomics', 'interactomics', 'phosphoproteomics']


class DataUploadApp(basicApp.BasicApp):
    """
    Defines what the dataUpload App is in the report_manager.
    Used to upload experimental and clinical data to correct project folder.

    .. warning:: There is a size limit of 55MB. Files bigger than this will have to be moved manually.
    """
    def __init__(self, title, subtitle, description, layout=[], logo=None, footer=None):
        self.pageType = "UploadDataPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        """
        Builds page with the basic layout from *basicApp.py* and adds relevant Dash components for project data upload.
        """
        self.add_basic_layout()
        layout = [html.Div([
                            html.Div([html.H4('Project identifier:', style={'marginTop': 30, 'marginBottom': 20}),
                                      dcc.Input(id='project_id', placeholder='e.g. P0000001', type='text', value='', debounce=True, maxLength=8, minLength=8, style={'width':'100%', 'height':'55px'}),
                                      dcc.Markdown(id='existing-project')],
                                     style={'width': '20%'}),
                            html.Br(),
                            html.Div(id='upload-form', children=[
                                html.Div(children=[html.A("Download example files",
                                                          id='example_files',
                                                          href= '/example_files',
                                                          n_clicks=0,
                                                          className="button_link")],
                                         style={'width':'100%', 'padding-left': '87%', 'padding-right': '0%'}),
                                html.Div(children=[html.Label('Select upload data type:', style={'marginTop': 10})],
                                               style={'width': '49%', 'marginLeft': '0%', 'verticalAlign': 'top', 'fontSize': '18px'}),
                                html.Div(children=[dcc.RadioItems(id='upload-data-type-picker', options=[{'label': i, 'value': i} for i in DataTypes], value=None,
                                                              inputStyle={"margin-right": "5px"}, style={'display': 'block', 'fontSize': '16px'})]),
                                html.Div(children=[html.H5('Proteomics tool:'), dcc.RadioItems(id='prot-tool', options=[{'label': i, 'value': i} for i in ['MaxQuant', 'DIA-NN','Spectronaut', 'FragPipe', 'mzTab']], value='',
                                                              inputStyle={"margin-right": "5px"}, style={'display': 'block', 'fontSize': '16px'})], id='proteomics-tool', style={'padding-top': 20}),
                                html.Div(children=[html.H5('Select the type of file uploaded:'), dcc.Dropdown(id='prot-file', options=[{'label': i, 'value': i} for i in ['Protein groups', 'Peptides', 'Phospho STY sites']], value='',
                                                              style={'display': 'block', 'fontSize': '14px', 'width': '250px'})], id='proteomics-file', style={'padding-top': 20}),
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
                                html.Div(children=[dcc.Markdown('**Uploaded Files:**', id='markdown-title'), dcc.Markdown(id='uploaded-files')]),
                                html.Div([html.Button("Upload Data to CKG",
                                             id='submit_button',
                                             n_clicks=0,
                                             className="button_link")],
                                      style={'width':'100%', 'padding-left': '87%', 'padding-right': '0%'})]),
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

        self.extend_layout(layout)
