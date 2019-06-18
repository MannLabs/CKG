import os.path
import plotly.io as pio
import h5py as h5
import json
import natsort
import plotly.utils
import plotly.graph_objs as go
from plotly.offline import iplot
from collections import defaultdict
from IPython.display import IFrame, display
import tempfile
import networkx as nx
from networkx.readwrite import json_graph
from report_manager import utils
from report_manager.plots import basicFigures

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

    def print_report(self, directory, plot_format='pdf'):
        for plot_id in self._plots:
            name = "_".join(plot_id)
            for plot in self._plots[plot_id]:
                figure = plot.figure
                pio.write_image(figure, os.path.join(directory,name+"."+plot_format))

    def save_report(self, directory):
        dt = h5.special_dtype(vlen=str)
        with h5.File(os.path.join(directory, "report.h5"), "w") as f:
            order = 0
            markdown = utils.convert_dash_to_json(utils.get_markdown_date("Report created on:"))
            figure_json = json.dumps(markdown, cls=utils.NumpyEncoder)
            figure_id = str(order) +"_date"
            grp = f.create_group(str(order) +"_date")
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
                            figure_json['notebook'] = plot['net_json']
                        if 'app' in plot:
                            json_str = utils.convert_dash_to_json(plot['app'])
                            figure_json['app'] = json_str
                        if 'net_tables' in plot:
                            json_str_nodes = utils.convert_dash_to_json(plot['net_tables'][0])
                            json_str_edges = utils.convert_dash_to_json(plot['net_tables'][1])
                            figure_json["net_tables"] = (json_str_nodes,json_str_edges)
                        figure_json = json.dumps(figure_json, cls=utils.NumpyEncoder)
                        figure_id = str(i)+'_net'
                    else:
                        json_str = utils.convert_dash_to_json(plot)
                        figure_json = json.dumps(json_str, cls=utils.NumpyEncoder)
                        figure_id = str(i) + '_figure' 
                    i += 1
                    fig_set = grp.create_dataset(figure_id, (1,), dtype=dt)
                    fig_set[:] = str(figure_json)
                    fig_set.attrs['identifier'] = figure_id
                order += 1

    #ToDo load Network data
    def read_report(self, directory):
        report_plots = defaultdict(list)
        with h5.File(os.path.join(directory, "report.h5"), 'r') as f:
            for name in f:
                plot_id = name.split('~')
                if len(plot_id) >1:
                    analysis = plot_id[0]
                    plot_type = plot_id[1]
                for figure_id in f[name]:
                    figure_json = f[name+"/"+figure_id][0]
                    identifier = f[name+"/"+figure_id].attrs["identifier"]
                    if 'net' in identifier:
                        figure = {}
                        net_json = json.loads(figure_json)
                        if 'notebook' in net_json:
                            figure['net_json'] = net_json['notebook']
                            netx = json_graph.node_link_graph(figure['net_json'])
                            figure['notebook'] = basicFigures.get_notebook_network_pyvis(netx)
                        if 'app' in net_json:
                            figure['app'] = net_json['app']
                    else:
                        figure = json.loads(figure_json)
                    report_plots[name].append(figure)
        self.plots = report_plots

    def visualize_report(self, environment):
        report_plots = []
        for plot_type in natsort.natsorted(self.plots):
            for plot in self.plots[plot_type]:
                if environment == "notebook":
                    if "notebook" in plot:
                        net = plot['notebook']
                        if not os.path.isdir('./tmp'):
                            os.makedirs('./tmp')
                        fnet = tempfile.NamedTemporaryFile(suffix=".html", delete=False, dir='tmp/')
                        with open(fnet.name, 'w') as f:
                            f.write(net.html)
                        display(IFrame(os.path.relpath(fnet.name),width=800, height=850))
                    else:
                        if 'props' in plot:
                            if 'figure' in plot['props']:
                                try:
                                    iplot(plot['props']['figure'])
                                except:
                                    pass
                         
                else:
                    if isinstance(plot, dict):
                        if "app" in plot:
                            plot = plot["app"]
                        if 'net_tables' in plot:
                            tables = plot['net_tables']
                            app_plots.append(tables[0])
                            app_plots.append(tables[1])

                    report_plots.append(plot)

        return report_plots

    def download_report(self, directory):
        for plot_type in natsort.natsorted(self.plots):
            name = "_".join(plot_type) if isinstance(plot_type, tuple) else plot_type
            print(name)
            for plot in self.plots[plot_type]:
                if "net_json" in plot:
                    with open(os.path.join(directory, name+'.json'), 'w') as out:
                        out.write(plot["net_json"])
                    
                    graph = json_graph.node_link_graph(plot["net_json"])
                    try:
                        nx.write_gml(graph, os.path.join(directory, name+".gml"))
                    except: 
                        pass
                    if "app" in plot:
                        plot = plot["app"]
                if 'props' in plot:
                    if 'figure' in plot['props']:
                        try:
                            basicFigures.save_DASH_plot(plot['props']['figure'], name=name, plot_format='svg', directory=directory)
                        except:
                            pass
                else:
                    try:
                        basicFigures.save_DASH_plot(plot.figure, name=name, plot_format='svg', directory=directory)
                    except:
                        pass
