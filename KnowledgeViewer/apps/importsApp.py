import dash
import dash_html_components as html
from dash.dependencies import Input, Output
from plots import basicFigures
import pandas as pd
import dash_auth
from apps import basicApp
from apps import imports
from KnowledgeViewer.queries import project_cypher
from KnowledgeViewer.viewer import viewer
from KnowledgeViewer import project


class ImportsApp(basicApp.BasicApp):
    def __init__(self, title, subtitle, description, layout = [], logo = None, footer = None):
        self.pageType = "importsPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        self.addBasicLayout()
        conf = self.getPageConfiguration()
        stats_file = conf['stats_file']
        stats_df = imports.get_stats_data(stats_file)
        plots = []
        plots.append(imports.plot_total_number_imported(stats_df, 'Number of imported entities and relationships'))
        plots.append(imports.plot_total_numbers_per_date(stats_df, 'Imported entities vs relationships'))
        plots.append(imports.plot_databases_numbers_per_date(stats_df, 'Imported entities/relationships per database', dropdown=True, dropdown_options='dates'))
        plots.append(imports.plot_import_numbers_per_database(stats_df, 'Breakdown imported entities/relationships', subplot_titles = ('Entities imported', 'Relationships imported', 'File size', 'File size'), colors=True, colors_1='entities', colors_2='relationships', dropdown=True, dropdown_options='all'))
        self.extendLayout(plots)
