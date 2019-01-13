from apps import basicApp
from report_manager import project
import dash_html_components as html
import dash_core_components as dcc


class ProjectApp(basicApp.BasicApp):
    def __init__(self, projectId, title, subtitle, description, layout = [], logo = None, footer = None):
        self._project_id = projectId
        self._page_type = "projectPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.page_type, layout, logo, footer)
        self.build_page()
    
    @property
    def project_id(self):
        return self._project_id

    @project_id.setter
    def project_id(self, project_id):
        self._project_id = project_id

    def build_page(self):
        p = project.Project(self.project_id, datasets=None, report={})
        self.title = "Project: {}".format(p.name)
        self.add_basic_layout()
        plots = p.show_report("app")
        tabs = []
        for data_type in plots:
            tab = dcc.Tab(label=data_type, children=[html.Div(plots[data_type])])
            tabs.append(tab)
        lc = dcc.Tabs(id="tabs", children=tabs)
        self.add_to_layout(lc)
        
        
