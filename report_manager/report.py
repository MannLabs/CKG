import os.path
import plotly.io as pio
import h5py as h5
import json
import plotly.utils
from plotly.offline import iplot
from collections import defaultdict
from networkx.readwrite import json_graph
from report_manager import utils

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
            for plot_id in self.plots:
                name = "~".join(plot_id)
                grp = f.create_group(name)
                i = 0
                for plot in self._plots[plot_id]:       
                    figure_id = None
                    if isinstance(plot, dict):
                        if 'net_json' in plot:
                            figure_json = json.dumps(plot['net_json'])
                            figure_id = 'net_'+str(i)
                    else:
                        json_str = utils.convert_dash_to_json(plot)
                        figure_json = json.dumps(json_str, cls=utils.NumpyEncoder)
                        figure_id = 'figure_' + str(i)
                    i += 1
                    fig_set = grp.create_dataset(figure_id, (1,), dtype=dt)
                    fig_set[:] = str(figure_json)
                    fig_set.attrs['identifier'] = figure_id

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
                    if identifier == name+'~net':
                        continue
                        json_graph = json.loads(figure_json)
                        net = json_graph.node_link_graph(json.loads(json_graph))
                        cy_elements = utils.networkx_to_cytoscape(net)
                    else:
                        figure = json.loads(figure_json)
                    report_plots[name].append(figure)
        self.plots = report_plots

    def visualize_report(self, environment):
        report_plots = defaultdict(list)
        
        for plot_type in self.plots:
            print(plot_type)
            for plot in self.plots[plot_type]:
                if environment == "notebook":
                    if "notebook" in plot:
                        net = plot['notebook']
                        if not os.path.isdir('./tmp'):
                            os.makedirs('./tmp')
                            fnet = tempfile.NamedTemporaryFile(suffix=".html", delete=False, dir='tmp/')
                            with open(fnet.name, 'w') as f:
                                f.write(net.html)
                            display(IFrame(os.path.relpath(fnet.name),width=1400, height=1400))
                            if hasattr(plot["net_tables"][0], 'figure') and hasattr(plot["net_tables"][1], 'figure'):
                                iplot(plot["net_tables"][0].figure)
                                iplot(plot["net_tables"][1].figure)
                    else:
                        if 'props' in plot:
                            if 'figure' in plot['props']:
                                try:
                                    iplot(plot['props']['figure'])
                                except:
                                    pass
                else:
                    report_plots[identifier].append(plot)

        return report_plots
