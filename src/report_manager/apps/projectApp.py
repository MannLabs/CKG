import os
from apps import basicApp
from report_manager import project
import dash_html_components as html
import dash_core_components as dcc


class ProjectApp(basicApp.BasicApp):
    """
    Defines what a project App is in the report_manager.
    Includes multiple tabs for different data types.
    """
    def __init__(self, projectId, title, subtitle, description, layout = [], logo = None, footer = None):
        self._project_id = projectId
        self._page_type = "projectPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.page_type, layout, logo, footer)
        self.build_page()

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

    def build_page(self):
        """
        Builds project and generates the report.
        For each data type in the report (e.g. 'proteomics', 'clinical'), \
        creates a designated tab.
        A button to download the entire project and report is added.
        """
        p = project.Project(self.project_id, datasets={}, knowledge=None, report={})
        p.build_project()
        p.generate_report()
        
        if p.name is not None:
            self.title = "Project: {}".format(p.name)
        else:
            self.title = ''
        self.add_basic_layout()
        
        plots = p.show_report("app")
        
        tabs = []
        button = html.Div([html.A('Download Project',
                                id='download-zip',
                                href="",
                                target="_blank",
                                n_clicks = 0
                                )
                            ])
        self.add_to_layout(button)
        for data_type in plots:
            if len(plots[data_type]) >=1:
                tab_content = [html.Div(plots[data_type])]
                tab = dcc.Tab(tab_content, label=data_type)
                tabs.append(tab)
        lc = dcc.Tabs(tabs)
        self.add_to_layout(lc)
