import os
import ckg_utils
import config.ckg_config as ckg_config
from report_manager import analysisResult as ar, report as rp
from report_manager.analyses import basicAnalysis
from graphdb_connector import connector
import logging
import logging.config

log_config = ckg_config.report_manager_log
logger = ckg_utils.setup_logging(log_config, key="dataset")

class Dataset:
    def __init__(self, identifier, dtype, configuration, data, analyses):
        self.identifier = identifier
        self.dataset_type = dtype
        self.configuration = configuration
        self.data = data
        self.analyses = analyses
        if len(data) == 0:
            self.data = self.query_data()
    
    @property
    def identifier(self):
        return self.identifier

    @identifier.setter
    def identifier(self, identifier):
        self.identifier = identifier

    @property
    def dataset_type(self):
        return self.dataset_type

    @dataset_type.setter
    def dataset_type(self, dataset_type):
        self.dataset_type = dataset_type

    @property
    def data(self):
        return self.data

    @data.setter
    def data(self, data):
        self.data = data

    @property
    def analyses(self):
        return self.analyses

    @analyses.setter
    def analyses(self, analyses):
        self.analyses = analyses

    @property
    def configuration(self):
        return self.configuration

    @configuration.setter
    def configuration(self, configuration):
        self.configuration = configuration
    
    def get_dataset(self, dataset):
        if dataset in self.data:
            return self.data[dataset]
        return None
    
    def get_analysis(self, analysis):
        if analysis in self.analyses:
            return self.analyses[analysis]
        return None

    def updateData(self, new):
        self.data.update(new)

    def updateAnalyses(self, new):
        self.analyses.update(new)

    def query_data(self):
        data = {}
        driver = connector.getGraphDatabaseConnectionConfiguration()
        replace = [("PROJECTID", self.identifier)]
        try:
            cwd = os.path.abspath(os.path.dirname(__file__))
            queries_path = "queries/datasets_cypher.yml"
            datasets_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
            if "replace" in self.configuration:
                replace = self.configuration["replace"]
            for query_name in datasets_cypher[self.dataset_type]:
                title = query_name.lower().replace('_',' ')
                query = datasets_cypher[self.dataset_type][query_name]['query']
                for r,by in replace:
                    query = query.replace(r,by)
                data[title] = connector.getCursorData(driver, query)
        except Exception as err:
            logger.error("Reading queries > {}.".format(err))
        
        return data

    def generateReport(self):
        report = rp.Report(self.dataset_type.capitalize())
        for key in self.configuration:
            for section_query,analysis_types,plot_names,args in self.configuration[key]:
                if section_query in self.data:
                    data = self.data[section_query]
                    result = None 
                    if len(analysis_types) >= 1:
                        for analysis_type in analysis_types:
                            result = ar.AnalysisResult(self.identifier, analysis_type, args, data)
                            self.updateAnalyses(result.getResult())
                            if key == "regulation":
                                reg_data = result.getResult()[analysis_type]
                                if not reg_data.empty:
                                    sig_hits = list(set(reg_data.loc[reg_data.rejected,"identifier"]))
                                    #sig_names = list(set(reg_data.loc[reg_data.rejected,"name"]))
                                    sig_data = data[sig_hits]
                                    sig_data.index = data['group'].tolist()
                                    self.updateData({"regulated":sig_data})
                            for plot_name in plot_names:
                                plots = result.getPlot(plot_name, section_query+"_"+analysis_type+"_"+plot_name, analysis_type.capitalize())
                                report.updatePlots({(analysis_type,plot_name):plots})
                    else:
                        if result is None:
                            dictresult = {}
                            dictresult["_".join(section_query.split(' '))] = data
                            result = ar.AnalysisResult(self.identifier,"_".join(section_query.split(' ')), {}, data, result = dictresult)
                            self.updateAnalyses(result.getResult())
                        for plot_name in plot_names:
                            plots = result.getPlot(plot_name, "_".join(section_query.split(' '))+"_"+plot_name, section_query.capitalize())
                            report.updatePlots({("_".join(section_query.split(' ')),plot_name):plots})
        return report

class ProteomicsDataset(Dataset):
    def __init__(self, identifier, configuration, data={}, analyses={}):
        Dataset.__init__(self, identifier, "proteomics", configuration, data, analyses)
        self.preprocessDataset()
        
    def preprocessDataset(self):
        processed_data = self.preprocessing()
        self.updateData({"preprocessed":processed_data})
    
    def preprocessing(self):
        processed_data = None
        data = self.get_dataset("dataset")
        if data is not None:
            imputation = True
            method = "mixed"
            missing_method = 'percentage'
            missing_max = 0.3
            value_col = 'LFQ intensity'
            args = {}
            if "args" in self.configuration:
                args = self.configuration["args"] 
            if "imputation" in args:
                imputation = args["imputation"]
            if "imputation_method" in args:
                method = args["imputation_method"]
            if "missing_method" in args:
                missing_method = args["missing_method"]
            if "missing_max" in args:
                missing_max = args["missing_max"]
            if "value_col" in args:
                value_col = args["value_col"]
            
            processed_data = basicAnalysis.get_measurements_ready(data, imputation = imputation, method = method, missing_method = missing_method, missing_max = missing_max)
        return processed_data

class WESDataset(Dataset):
    def __init__(self, identifier, configuration, data={}):
        Dataset.__init__(identifier, "wes", configuration, data)
