import dash
import dash_html_components as html
from dash.dependencies import Input, Output
from plots import basicFigures
import pandas as pd
import dash_auth
from apps import basicApp
from KnowledgeViewer.queries import project_cypher
from KnowledgeViewer.viewer import viewer
from KnowledgeViewer import project


class ProjectApp(basicApp.BasicApp):
    def __init__(self, projectId, title, subtitle, description, layout = [], logo = None, footer = None):
        self.projectId = projectId
        self.pageType = "projectPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def setProjectId(self, projectId):
        self.projectId = projectId

    def getProjectId(self):
        return self.projectId

    def buildPage(self):
        self.addBasicLayout()
        p = project.Project(self.getProjectId(), 'multi-omics')
        plots = p.showReport("app")
        self.extendLayout(plots)
        
        
