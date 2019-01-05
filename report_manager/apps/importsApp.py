import config.ckg_config as ckg_config
from apps import basicApp
from apps import imports


class ImportsApp(basicApp.BasicApp):
    def __init__(self, title, subtitle, description, layout = [], logo = None, footer = None):
        self.pageType = "importsPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        self.add_basic_layout()
        stats_file = "../../data/imports/stats/stats.hdf"
        stats_key =  'full_stats_'+ str(ckg_config.version).replace('.','_')
        stats_df = imports.get_stats_data(stats_file, key=stats_key)
        plots = []
        plots.append(imports.plot_total_number_imported(stats_df, 'Number of imported entities and relationships'))
        plots.append(imports.plot_total_numbers_per_date(stats_df, 'Imported entities vs relationships'))
        plots.append(imports.plot_databases_numbers_per_date(stats_df, 'Imported entities/relationships per database', dropdown=True, dropdown_options='dates'))
        plots.append(imports.plot_import_numbers_per_database(stats_df, 'Breakdown imported entities/relationships', subplot_titles = ('Entities imported', 'Relationships imported', 'File size', 'File size'), colors=True, colors_1='entities', colors_2='relationships', dropdown=True, dropdown_options='all'))
        self.extend_layout(plots)
