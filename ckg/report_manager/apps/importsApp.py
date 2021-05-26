import os
from ckg import ckg_utils
from ckg.report_manager.apps import basicApp
from ckg.report_manager.apps import imports
from ckg.analytics_core.viz import viz


class ImportsApp(basicApp.BasicApp):
    """
    Defines what the imports App is in the report_manager.
    Enables the tracking of the number of imported entitites and relationships, updates and file sizes.
    """
    def __init__(self, title, subtitle, description, layout = [], logo = None, footer = None):
        self.pageType = "importsPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        """
        Builds page with the basic layout from *basicApp.py* and adds all the relevant plots from *imports.py*.
        """
        plots = []
        self.add_basic_layout()
        stats_dir = ckg_utils.read_ckg_config(key='stats_directory')
        stats_file = os.path.join(stats_dir, "stats.hdf")
        if os.path.exists(stats_file):
            stats_df = imports.get_stats_data(stats_file, n=3)
            plots.append(imports.plot_total_number_imported(stats_df, 'Number of imported entities and relationships'))
            plots.append(imports.plot_total_numbers_per_date(stats_df, 'Imported entities vs relationships'))
            plots.append(imports.plot_databases_numbers_per_date(stats_df, 'Full imports: entities/relationships per database', key='full', dropdown=True, dropdown_options='dates'))
            plots.append(imports.plot_databases_numbers_per_date(stats_df, 'Partial imports: entities/relationships per database', key='partial', dropdown=True, dropdown_options='dates'))
            plots.append(imports.plot_import_numbers_per_database(stats_df, 'Full imports: Breakdown entities/relationships', key='full', subplot_titles = ('Entities imported', 'Relationships imported', 'File size', 'File size'), colors=True, plots_1='entities', plots_2='relationships', dropdown=True, dropdown_options='databases'))
            plots.append(imports.plot_import_numbers_per_database(stats_df, 'Partial imports: Breakdown entities/relationships', key='partial', subplot_titles = ('Entities imported', 'Relationships imported', 'File size', 'File size'), colors=True, plots_1='entities', plots_2='relationships', dropdown=True, dropdown_options='databases'))
        else:
            plots.append(viz.get_markdown(text="# There are no statistics about recent imports."))

        self.extend_layout(plots)
