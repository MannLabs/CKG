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
        statsfile = conf['stats_file']
        stats_df = imports.get_stats_data(stats_ile)
        plots = []
        plots.append(imports.plot_total_number_imported(stats_df, '', 'Date', 'Number of imports'))
        plots.append(imports.plot_total_numbers_per_date(stats_df, '', 'Date', 'Number of imports'))
        plots.append(imports.plot_databases_numbers_per_date(stats_df, '', '', '', dropdown=True, dropdown_options='dates'))
        plots.append(imports.plot_import_numbers_per_database(stats_df, '', '', '', subplot_titles = ('Entities imports', 'Entities file size','Relationships imports', 'Relationships file size'), colors=True, colors_1='entities', colors_2='relationships', dropdown=True, dropdown_options='all'))
        self.extendLayout(plots)
