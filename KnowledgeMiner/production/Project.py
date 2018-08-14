import project_config as config
import project_cypher as cypher

class Project:
    def __init__(identifier, project_type, datasets = []):
        self.identifier = identifier
        self.project_type = project_type
        self.datasets = datasets
        if len(datasets) == 0:
            self.buildProject()

    def getIdentifier(self):
        return self.identifier

    def getProject_type(self):
        return self.project_type

    def getDatasets(self):
        return self.datasets

    def setIdentifier(self, identifier):
        self.identifier = identifier

    def setProject_type(self, project_type):
        self.project_type = project_type

    def setDatasets(self, datasets):
        self.datasets = datasets

    def buildProject(self):
        for dataset_type in config:
            if dataset_type == "proteomics":
                proteomicsDataset = ProteomicsDataset(self.getIdentifier(), config[dataset_type])



            for section in config[key]:
                for section_query,analysis_types,plot_names,args in config[key][section]:
                    args["id"] = self.getProjectId()
                    plots = viewer.view(key, section_query, analysis_types, plot_names, args)
                    self.extendLayout(plots)

    def generateReport(self, configuration):
        for dataset in self.getDatasets():
            if dataset.getType() in configuration: 
                dataset.generateReport(configuration[dataset.getType()])
    
class ProteomicsDataset:
    def __init__(identifier, configuration, data={}):
        self.identifier = identifier
        self.type = "proteomics"
        self.configuration = configuration
        self.data = data
        if len(data) == 0:
            self.queryData()

    def getIdentifier(self):
        return self.identifier

    def getType(self):
        return self.type

    def getConfiguration(self):
        return self.configuration

    def setIdentifier(self, identifier):
        self.identifier = identifier

    def setType(self):
        self.type = "proteomics"

    def setConfiguration(self, configuration):
        self.configuration  = configuration

    def preprocessData(data, args):
        imputation = True
        method = "mixed"
        missing_method = 'percentage'
        missing_max = 0.3
        
        if "imputation" in args:
            imputation = args["imputation"]
        if "imputation_method" in args:
            method = args["imputation_method"]
        if "missing_method" in args:
            missing_method = args["missing_method"]
        if "missing_max" in args:
            missing_max = args["missing_max"]
        
        data = analyses.get_measurements_ready(data, imputation = imputation, method = method, missing_method = missing_method, missing_max = missing_max)
    return data

