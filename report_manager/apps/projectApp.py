from apps import basicApp
from report_manager import project


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
        self.add_basic_layout()
        p = project.Project(self.project_id)
        plots = p.show_report("app")
        self.extend_layout(plots)
        
        
