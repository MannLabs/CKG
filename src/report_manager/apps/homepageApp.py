import config.ckg_config as ckg_config
from apps import basicApp
from apps import homepageStats as hpstats


class HomePageApp(basicApp.BasicApp):
    """
    Defines what the HomePage App is in the report_manager.
    Enables the tracking of the number of entitites, relationships, projects, and others, currently in the grapha database.
    """
    def __init__(self, title, subtitle, description, layout = [], logo = None, footer = None):
        self.pageType = "homePage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        """
        Builds page with the basic layout from *basicApp.py* and adds all the relevant plots from *homepageStats.py*.
        """
        self.add_basic_layout()
        layout = hpstats.quick_numbers_panel()
        dfs = hpstats.get_db_stats_data()
        plots = []
        plots.append(hpstats.plot_store_size_components(dfs, title='DB Store Size'))
        plots.append(hpstats.plot_node_rel_per_label(dfs, focus='nodes', title='Nodes per Label'))
        plots.append(hpstats.plot_node_rel_per_label(dfs, focus='relationships', title='Relationships per Type'))
        self.extend_layout(layout)
        self.extend_layout(plots)