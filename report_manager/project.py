from report_manager import project_config as config
from report_manager.dataset import ProteomicsDataset
from graphdb_connector import connector
from plotly.offline import iplot

class Project:
    ''' A project class that defines an experimental project.
        A project can be of different types, contain several datasets and reports.
        To use:
         >>> p = Project(identifier="P0000001", project_type="multi-omics", datasets=None, report=None)
         >>> p.showReport(environment="noteboook")
    '''

    def __init__(self, identifier, datasets = None, report = None):
        self.identifier = identifier
        self.project_type = project_type
        self.datasets = datasets
        self.report = {}
        if self.datasets is None:
            self.datasets = {}
            self.buildProject()
            self.generateReport()

    @property
    def identifier(self):
        return self.identifier
    
    @identifier.setter
    def identifier(self, identifier):
        self.identifier = identifier

    @property
    def project_type(self):
        return self.project_type

    @project_type.setter
    def project_type(self, project_type):
        self.project_type = project_type
    
    @property
    def datasets(self):
        return self.datasets

    @datasets.setter
    def datasets(self, datasets):
        self.datasets = datasets

    @property
    def report(self):
        return self.report

    @report.setter
    def report(self, report):
        self.report = report
    
    def getDataset(self, dataset):
        if dataset in self.datasets:
            return self.datasets[dataset]
        return None

    def updateDataset(self, dataset):
        self.datasets.update(dataset)

    def updateReport(self, new):
        self.report.update(new)

    def buildProject(self):
        for dataset_type in config.configuration:
            if dataset_type == "proteomics":
                proteomicsDataset = ProteomicsDataset(self.getIdentifier(), config.configuration[dataset_type])
                self.updateDataset({dataset_type:proteomicsDataset})
           
    def generateReport(self):
        if len(self.report) == 0:
            for dataset_type in self.getDatasets():
                dataset = self.getDataset(dataset_type)
                if dataset is not None:
                    report = dataset.generateReport()
                    self.updateReport({dataset.getType():report})
    
    def emptyReport(self):
        self.report = {}

    def generateDatasetReport(self, dataset):
        if dataset_type in self.datasets:
            dataset = self.getDataset(dataset_type)
            if dataset is not None:
                        report = dataset.generateReport()
                        self.updateReport({dataset.getType():report})

    def showReport(self, environment):
        app_plots = []
        for data_type in self.getReport():
            plots = self.getReport()[data_type].getPlots()
            for plot_type in plots:
                for plot in plots[plot_type]:
                    if environment == "notebook":
                        iplot(plot.figure)
                    else:
                        app_plots.append(plot)

        return app_plots        
