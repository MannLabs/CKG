from plotly.offline import iplot
from json import dumps
import ckg_utils
import config.ckg_config as ckg_config
from report_manager.dataset import ProteomicsDataset
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

    def __init__(self, identifier, datasets = None, report = None):
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
            attributes = project_info["attributes"]
            if "name" in attributes:
                self.name = attributes["name"]
            if "acronym" in attributes:
                self.acronym = attributes["acronym"]
            if "description" in attributes:
                self.description = attributes["description"]
            if "data_types" in attributes:
                self.data_types = attributes["data_types"]
            if "responsible" in attributes:
                self.responsible = attributes["responsible"]
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
            "number_subjects": self.number_subjects
            }
        
        return d
        
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
            logger.error("Reading queries > {}.".format(err))
        
        return data

    def build_project(self):
        project_info = self.query_data()
        self.set_attributes(project_info)
        for data_type in self.data_types:
            if data_type == "proteomics":
                proteomics_dataset = ProteomicsDataset(self.identifier)
                self.update_dataset({dataset_type:proteomics_dataset})
           
    def generate_project_info_report(self):
        report = rp.Report("project_info")
        project_dict = self.to_dict()
        identifier = "project_info"
        title = "Project: {} information".format(self.name)
        plot = [figure.getBasicTable(project_dict, identifier, title)]
        report.update_plots({("project_info","Project Information"): plot})
        
        return report
    
    def generate_report(self):
        if len(self.report) == 0:
            project_report = self.generate_project_info_report()
            self.update_report(report)
            for dataset_type in self.datasets:
                dataset = self.get_dataset(dataset_type)
                if dataset is not None:
                    report = dataset.generate_report()
                    self.update_report({dataset.project_type:report})
    
    def empty_report(self):
        self.report = {}

    def generate_dataset_report(self, dataset):
        dataset = self.get_dataset(dataset_type)
        if dataset is not None:
            report = dataset.generate_report()
            self.update_report({dataset.project_type:report})

    def show_report(self, environment):
        app_plots = []
        for data_type in self.report:
            plots = self.report[data_type].plots()
            for plot_type in plots:
                for plot in plots[plot_type]:
                    if environment == "notebook":
                        iplot(plot.figure)
                    else:
                        app_plots.append(plot)

        return app_plots        
