import os
from datetime import datetime
from uuid import uuid4
from ckg import ckg_utils
from ckg.report_manager.apps import basicApp
from ckg.report_manager import project
import dash_html_components as html
import dash_core_components as dcc
from ckg.report_manager.worker import generate_project_report


class ProjectApp(basicApp.BasicApp):
    """
    Defines what a project App is in the report_manager.
    Includes multiple tabs for different data types.
    """
    def __init__(self, id, projectId, title, subtitle, description, layout=[], logo=None, footer=None, force=False):
        self._id = id
        print("Project id", self._id)
        self._project_id = projectId
        self._page_type = "projectPage"
        self._session_id = projectId + datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        self._force = force
        self._configuration_files = {}
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.page_type, layout, logo, footer)
        self.build_page()

    @property
    def id(self):
        """
        Retrieves page identifier.
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets 'id' input value as id property of the class.

        :param str id: page identifier.
        """
        self._id = id

    @property
    def session_id(self):
        """
        Retrieves session identifier.
        """
        return self._session_id

    @session_id.setter
    def session_id(self, session_id):
        """
        Sets 'session_id' input value as id property of the class.

        :param str session_id: session identifier.
        """
        self._session_id = session_id

    @property
    def project_id(self):
        """
        Retrieves project identifier.
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id):
        """
        Sets 'project_id' input value as project_id property of the class.

        :param str project_id: project identifier.
        """
        self._project_id = project_id

    @property
    def configuration_files(self):
        """
        Retrieves project configuration files.
        """
        return self._configuration_files

    @configuration_files.setter
    def configuration_files(self, configuration_files):
        """
        Sets 'configuration_files' input value as configuration_files property of the class.

        :param dict configuration_files: configuration files.
        """
        self._configuration_files = configuration_files

    @property
    def force(self):
        """
        Retrieves attribute force (whether or not the project report needs to be regenerated).
        """
        return self._force

    @force.setter
    def force(self, force):
        """
        Sets 'force' value as force property of the class.

        :param boolean force: force.
        """
        self._force = force

    def build_header(self):
        buttons = html.Div([html.Div([html.A('Download Project Report',
                                id='download-zip',
                                href="",
                                target="_blank",
                                n_clicks=0,
                                className="button_link"
                                )]),
                            html.Div([html.A("Regenerate Project Report",
                                id='regenerate',
                                title=self.id,
                                href='',
                                target='',
                                n_clicks=0,
                                className="button_link")]),
                            html.Div([html.H3("Change Analysis' Configuration: "),
                            dcc.Dropdown(
                                id='my-dropdown',
                                options=[
                                    {'label': '', 'value': self.id+'/defaults'},
                                    {'label': 'Proteomics configuration', 'value': self.id+'/proteomics'},
                                    {'label': 'Interactomics configuration', 'value': self.id+'/interactomics'},
                                    {'label': 'Phosphoproteomics configuration', 'value': self.id+'/phosphoproteomics'},
                                    {'label': 'Clinical data configuration', 'value': self.id+'/clinical'},
                                    {'label': 'Multiomics configuration', 'value': self.id+'/multiomics'},
                                    {'label': 'Reset to defaults', 'value': self.id + '/reset'}],
                                value=self.id+'/defaults',
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

    def build_page(self):
        """
        Builds project and generates the report.
        For each data type in the report (e.g. 'proteomics', 'clinical'), \
        creates a designated tab.
        A button to download the entire project and report is added.
        """
        config_files = {}
        tmp_dir = ckg_utils.read_ckg_config(key='tmp_directory')
        if os.path.exists(tmp_dir):
            directory = os.path.join(tmp_dir, self.id)
            if os.path.exists(directory):
                config_files = {f.split('.')[0]: os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))}

        result = generate_project_report.apply_async(args=[self.project_id, config_files, self.force], task_id='generate_report'+self.session_id, queue='compute')
        result_output = result.get()
        print(result_output)

        p = project.Project(self.project_id, datasets={}, knowledge=None, report={}, configuration_files=config_files)
        p.build_project(False)
        
        if p.name is not None:
            self.title = "Project: {}".format(p.name)
        else:
            self.title = ''
        self.add_basic_layout()
        plots = p.show_report("app")
        p = None
        tabs = []
        buttons = self.build_header()

        self.add_to_layout(buttons)
        for data_type in plots:
            if len(plots[data_type]) >= 1:
                tab_content = [html.Div(plots[data_type])]
                tab = dcc.Tab(tab_content, label=data_type)
                tabs.append(tab)
        lc = dcc.Tabs(tabs)
        self.add_to_layout(lc)
