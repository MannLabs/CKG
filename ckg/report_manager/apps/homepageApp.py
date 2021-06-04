from ckg.report_manager.apps import basicApp
from ckg.report_manager.apps import homepageStats as hpstats


class HomePageApp(basicApp.BasicApp):
    """
    Defines what the HomePage App is in the report_manager.
    Enables the tracking of the number of entitites, relationships, projects, and others, currently in the grapha database.
    """
    def __init__(self, title, subtitle, description, layout=[], logo=None, footer=None):
        self.pageType = "homePage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        """
        Builds page with the basic layout from *basicApp.py* and adds all the relevant plots from *homepageStats.py*.
        """
        args = {}
        args['valueCol'] = 'value'
        args['textCol'] = 'size'
        args['y'] = 'index'
        args['x'] = 'number'
        args['orientation'] = 'h'
        args['title'] = ''
        args['x_title'] = ''
        args['y_title'] = ''
        args['height'] = 900
        args['width'] = 900

        self.add_basic_layout()
        layout = hpstats.quick_numbers_panel()
        dfs = hpstats.get_db_stats_data()
        plots = []
        plots.append(hpstats.plot_store_size_components(dfs, title='DB Store Size', args=args))
        plots.append(hpstats.plot_node_rel_per_label(dfs, focus='nodes', title='Nodes per Label', args=args))
        plots.append(hpstats.plot_node_rel_per_label(dfs, focus='relationships', title='Relationships per Type', args=args))
        self.extend_layout(layout)
        self.extend_layout(plots)
