import sys
import os
from collections import defaultdict
from plotly.offline import iplot
from IPython.display import IFrame, display
import tempfile
from json import dumps
import pandas as pd
import ckg_utils
import config.ckg_config as ckg_config
from report_manager.dataset import ProteomicsDataset, ClinicalDataset, DNAseqDataset, RNAseqDataset, LongitudinalProteomicsDataset
from report_manager.plots import basicFigures as figure
from report_manager import report as rp
from graphdb_connector import connector
import logging
import logging.config

log_config = ckg_config.report_manager_log
logger = ckg_utils.setup_logging(log_config, key="project")

class Project:
    ''' A project class that defines an experimental project.
        A project can be of different types, contain several datasets and reports.
        To use:
         >>> p = Project(identifier="P0000001", datasets=None, report=None)
         >>> p.show_report(environment="notebook")
    '''

    def __init__(self, identifier, datasets=None, report={}):
        self._identifier = identifier
        self._datasets = datasets
        self._report = report
        self._name = None
        self._acronym = None
        self._data_types = None
        self._responsible = None
        self._description = None
        self._status = None
        self._num_subjects = None
        if self._datasets is None:
            self._datasets = {}
            self.build_project()
            self.generate_report()

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        self._identifier = identifier

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def acronym(self):
        return self._acronym

    @acronym.setter
    def acronym(self, acronym):
        self._acronym = acronym

    @property
    def data_types(self):
        return self._data_types

    @data_types.setter
    def data_types(self, data_types):
        self._data_types = data_types

    @property
    def responsible(self):
        return self._responsible

    @responsible.setter
    def responsible(self, responsible):
        self._responsible = responsible

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    @property
    def num_subjects(self):
        return self._num_subjects

    @num_subjects.setter
    def num_subjects(self, num_subjects):
        self._num_subjects = num_subjects

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, datasets):
        self._datasets = datasets

    @property
    def report(self):
        return self._report

    @report.setter
    def report(self, report):
        self._report = report

    def get_dataset(self, dataset):
        if dataset in self.datasets:
            return self.datasets[dataset]
        return None

    def update_dataset(self, dataset):
        self.datasets.update(dataset)

    def update_report(self, new):
        self.report.update(new)

    def set_attributes(self, project_info):
        if "attributes" in project_info:
            attributes = project_info["attributes"].to_dict('r')[0]
            if "name" in attributes:
                self.name = attributes["name"]
            if "acronym" in attributes:
                self.acronym = attributes["acronym"]
            if "description" in attributes:
                self.description = attributes["description"]
            if "data_types" in attributes:
                self.data_types = attributes["data_types"].split(',')
            if "responsible" in attributes:
                self.responsible = attributes["responsible"].split(',')
            if "status" in attributes:
                self.status = attributes["status"]
            if "number_subjects" in attributes:
                self.num_subjects = attributes["number_subjects"]

    def to_dict(self):
        d = {"identifier" : self.identifier,
            "name" : self.name,
            "acronym" : self.acronym,
            "description" : self.description,
            "data_types" : self.data_types,
            "responsible": self.responsible,
            "status": self.status,
            "number_subjects": self.num_subjects
            }

        return d

    def to_dataframe(self):
        d = self.to_dict()
        df = pd.DataFrame.from_dict(d, orient='index')
        df = df.transpose()

        return df

    def to_json(self):
        d = self.to_dict()
        djson = dumps(d)

        return djson

    def query_data(self):
        data = {}
        driver = connector.getGraphDatabaseConnectionConfiguration()
        replace = [("PROJECTID", self.identifier)]
        try:
            cwd = os.path.abspath(os.path.dirname(__file__))
            queries_path = "queries/project_cypher.yml"
            project_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
            for query_name in project_cypher:
                title = query_name.lower().replace('_',' ')
                query = project_cypher[query_name]['query']
                for r,by in replace:
                    query = query.replace(r,by)
                data[title] = connector.getCursorData(driver, query)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))

        return data

    def build_project(self):
        project_info = self.query_data()
        self.set_attributes(project_info)
        for data_type in self.data_types:
            if data_type == "proteomics":
                dataset = ProteomicsDataset(self.identifier, data={}, analyses={}, analysis_queries={}, report=None)
                self.update_dataset({data_type:dataset})
            elif data_type == "clinical":
                dataset = ClinicalDataset(self.identifier, data={}, analyses={}, analysis_queries={}, report=None)
                self.update_dataset({data_type:dataset})
            elif data_type == "wes" or data_type == "wgs":
                dataset = DNAseqDataset(self.identifier, dataset_type=data_type, data={}, analyses={}, analysis_queries={}, report=None)
                self.update_dataset({data_type:dataset})
            elif data_type == "longitudinal_proteomics":
                dataset = LongitudinalProteomicsDataset(self.identifier, data={}, analyses={}, analysis_queries={}, report=None)
                self.update_dataset({data_type:dataset})


    def generate_project_info_report(self):
        report = rp.Report(identifier="project_info")
        project_df = self.to_dataframe()
        identifier = "Project info"
        title = "Project: {} information".format(self.name)
        plot = [figure.getBasicTable(project_df, identifier, title)]
        report.plots = {("Project info","Project Information"): plot}

        return report

    def generate_report(self):
        if len(self.report) == 0:
            project_report = self.generate_project_info_report()
            self.update_report({"Project information":project_report})
            for dataset_type in self.data_types:
                dataset = self.get_dataset(dataset_type)
                if dataset is not None:
                    dataset.generate_report()
                    self.update_report({dataset.dataset_type:dataset.report})

    def empty_report(self):
        self.report = {}

    def generate_dataset_report(self, dataset):
        dataset = self.get_dataset(dataset_type)
        if dataset is not None:
            report = dataset.generate_report()
            self.update_report({dataset.dataset_type:report})

    def show_report(self, environment):
        app_plots = defaultdict(list)
        for data_type in self.report:
            r = self.report[data_type]
            plots = r.plots
            identifier = r.identifier
            for plot_type in plots:
                for plot in plots[plot_type]:
                    if environment == "notebook":
                        if hasattr(plot, 'figure'):
                            iplot(plot.figure)
                        elif isinstance(plot, dict):
                            if "notebook" in plot:
                                net = plot['notebook']
                                fnet = tempfile.NamedTemporaryFile(suffix=".html", delete=False, dir='tmp/')
                                with open(fnet.name, 'w') as f:
                                    f.write(net.html)
                                display(IFrame(os.path.relpath(fnet.name),width=1400, height=1400))
                                iplot(plot["net_tables"][0].figure)
                                iplot(plot["net_tables"][1].figure)
                    else:
                        if isinstance(plot, dict):
                            if 'app' in plot:
                                app_plots[identifier].append(plot['app'])
                            if 'net_tables' in plot:
                                tables = plot['net_tables']
                                app_plots[identifier].append(tables[0])
                                app_plots[identifier].append(tables[1])
                        else:
                            app_plots[identifier].append(plot)

        return app_plots
