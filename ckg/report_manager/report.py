import os.path
import pandas as pd
import plotly.io as pio
import h5py as h5
import json
import natsort
import dash_html_components as html
from plotly.offline import iplot
from collections import defaultdict
from cyjupyter import Cytoscape
from ckg.report_manager import utils
from ckg import ckg_utils
from ckg.analytics_core.viz import viz
from ckg.analytics_core import utils as acore_utils


class Report:
    def __init__(self, identifier, plots={}):
        self._identifier = identifier
        self._plots = plots

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        self._identifier = identifier

    @property
    def plots(self):
        return self._plots

    @plots.setter
    def plots(self, plots):
        self._plots = plots

    def get_plot(self, plot):
        if plot in self._plots:
            return self._plots[plot]
        return None

    def update_plots(self, plot):
        self._plots.update(plot)

    def list_plots(self):
        plots = []
        if self.plots is not None:
            plots = self.plots.keys()

        return plots

    def print_report(self, directory, plot_format='pdf'):
        for plot_id in self._plots:
            name = "_".join(plot_id)
            for plot in self._plots[plot_id]:
                figure = plot.figure
                pio.write_image(figure, os.path.join(directory, name + "." + plot_format))

    def save_report(self, directory):
        dt = h5.special_dtype(vlen=str)
        with h5.File(os.path.join(directory, "report.h5"), "w") as f:
            order = 0
            markdown = ckg_utils.convert_dash_to_json(utils.get_markdown_date("Report created on:"))
            figure_json = json.dumps(markdown, cls=ckg_utils.NumpyEncoder)
            figure_id = str(order) + "_date"
            grp = f.create_group(str(order) + "_date")
            fig_set = grp.create_dataset(figure_id, (1,), dtype=dt)
            fig_set[:] = str(figure_json)
            fig_set.attrs['identifier'] = figure_id

            for plot_id in self.plots:
                name = "~".join(plot_id)
                grp = f.create_group(name)
                i = 0
                for plot in self._plots[plot_id]:
                    figure_id = None
                    if isinstance(plot, dict):
                        figure_json = {}
                        if 'net_json' in plot:
                            figure_json['net_json'] = plot['net_json']
                        if 'notebook' in plot:
                            figure_json['notebook'] = plot['notebook']
                        if 'app' in plot:
                            json_str = ckg_utils.convert_dash_to_json(plot['app'])
                            figure_json['app'] = json_str
                        if 'net_tables_viz' in plot:
                            json_str_nodes = ckg_utils.convert_dash_to_json(plot['net_tables_viz'][0])
                            json_str_edges = ckg_utils.convert_dash_to_json(plot['net_tables_viz'][1])
                            figure_json["net_tables_viz"] = (json_str_nodes, json_str_edges)
                        figure_json = json.dumps(figure_json, cls=ckg_utils.NumpyEncoder)
                        figure_id = str(i)+'_net'
                    else:
                        json_str = ckg_utils.convert_dash_to_json(plot)
                        figure_json = json.dumps(json_str, cls=ckg_utils.NumpyEncoder)
                        figure_id = str(i) + '_figure'
                    i += 1
                    fig_set = grp.create_dataset(figure_id, (1,), dtype=dt, compression="gzip")
                    try:
                        fig_set[:] = str(figure_json)
                        fig_set.attrs['identifier'] = figure_id
                    except ValueError as err:
                        print(figure_json, err)
                order += 1

    def read_report(self, directory):
        report_plots = defaultdict(list)
        if os.path.exists(os.path.join(directory, "report.h5")):
            with h5.File(os.path.join(directory, "report.h5"), 'r') as f:
                for name in f:
                    plot_id = name.split('~')
                    for figure_id in f[name]:
                        figure_json = f[name+"/"+figure_id][0]
                        identifier = f[name+"/"+figure_id].attrs["identifier"]
                        if 'net' in identifier:
                            figure = {}
                            net_json = json.loads(figure_json)
                            for key in net_json:
                                figure[key] = net_json[key]
                        else:
                            figure = json.loads(figure_json)
                        report_plots[name].append(figure)
        self.plots = report_plots

    def visualize_report(self, environment):
        report_plots = []
        for plot_type in natsort.natsorted(self.plots):
            for plot in self.plots[plot_type]:
                if plot is not None:
                    if environment == "notebook":
                        if "notebook" in plot:
                            net = plot['notebook']
                            net_viz = viz.visualize_notebook_network(net, notebook_type='jupyter',
                                                                     layout={'width': '100%', 'height': '700px'})
                            report_plots.append(net_viz)
                        else:
                            if isinstance(plot, dict):
                                if 'props' in plot:
                                    if 'figure' in plot['props']:
                                        try:
                                            iplot(plot['props']['figure'])
                                        except Exception:
                                            pass
                            elif hasattr(plot, 'figure'):
                                try:
                                    iplot(plot.figure)
                                except Exception:
                                    pass
                    else:
                        app_plot = plot
                        if isinstance(plot, dict):
                            if "app" in plot:
                                app_plot = plot["app"]
                            if 'net_tables_viz' in plot:
                                tables = plot['net_tables_viz']
                                report_plots.append(tables[0])
                                report_plots.append(tables[1])

                        report_plots.append(html.Div(app_plot, style={'overflowY': 'scroll', 'overflowX': 'scroll'}))

        return report_plots

    def visualize_plot(self, environment, plot_type):
        report_plots = []
        for plot in self.plots[plot_type]:
            if environment == "notebook":
                if "notebook" in plot:
                    net = plot['notebook']
                    report_plots = viz.visualize_notebook_network(net)
                else:
                    if 'props' in plot:
                        if 'figure' in plot['props']:
                            try:
                                iplot(plot['props']['figure'])
                            except Exception:
                                pass
            else:
                if isinstance(plot, dict):
                    if "app" in plot:
                        plot = plot["app"]
                    if 'net_tables_viz' in plot:
                        tables = plot['net_tables_viz']
                        report_plots.append(tables[0])
                        report_plots.append(tables[1])

                report_plots.append(plot)

        return report_plots

    def download_report(self, directory):
        saved = set()
        ckg_utils.checkDirectory(directory)
        for plot_type in natsort.natsorted(self.plots):
            name = "_".join(plot_type) if isinstance(plot_type, tuple) else plot_type
            i = 0
            for plot in self.plots[plot_type]:
                if plot is not None:
                    figure_name = name
                    if name in saved:
                        figure_name = name + "_" + str(i)
                        i += 1
                    if "net_json" in plot:
                        with open(os.path.join(directory, name+'.json'), 'w') as out:
                            out.write(json.dumps(plot["net_json"], cls=ckg_utils.NumpyEncoder))
                        try:
                            acore_utils.json_network_to_gml(plot["net_json"], os.path.join(directory, name+".gml"))
                        except Exception:
                            pass
                        if "net_tables" in plot:
                            nodes_table, edges_table = plot['net_tables']
                            if isinstance(nodes_table, pd.DataFrame):
                                nodes_table.to_csv(os.path.join(directory, name+'_node_table.tsv'), sep='\t', header=True, index=False, doublequote=False)
                            if isinstance(edges_table, pd.DataFrame):
                                edges_table.to_csv(os.path.join(directory, name+'_edges_table.tsv'), sep='\t', header=True, index=False, doublequote=False)
                        if "app" in plot:
                            plot = plot["app"]
                    if 'props' in plot:
                        if 'figure' in plot['props']:
                            try:
                                viz.save_DASH_plot(plot['props']['figure'], name=figure_name, plot_format='svg', directory=directory)
                                viz.save_DASH_plot(plot['props']['figure'], name=figure_name, plot_format='png', directory=directory)
                                saved.add(figure_name)
                            except Exception:
                                pass
                    else:
                        try:
                            viz.save_DASH_plot(plot.figure, name=figure_name, plot_format='svg', directory=directory)
                            viz.save_DASH_plot(plot.figure, name=figure_name, plot_format='png', directory=directory)
                            saved.add(figure_name)
                        except Exception:
                            pass
