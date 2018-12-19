import os
import sys
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
    def __init__(self, identifier, dtype, configuration=None, data={}, analyses={}):
        self._identifier = identifier
        self._dataset_type = dtype
        self._configuration = configuration
        self._data = data
        self._analyses = analyses
    
    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        self._identifier = identifier

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, dataset_type):
        self._dataset_type = dataset_type

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def analyses(self):
        return self._analyses

    @analyses.setter
    def analyses(self, analyses):
        self._analyses = analyses

    @property
    def configuration(self):
        return self._configuration

    @configuration.setter   
    def configuration(self, configuration):
        self._configuration = configuration
    
    def get_dataset(self, dataset):
        if dataset in self.data:
            return self.data[dataset]
        return None
    
    def get_analysis(self, analysis):
        if analysis in self.analyses:
            return self.analyses[analysis]
        return None

    def update_data(self, new):
        self.data.update(new)

    def update_analyses(self, new):
        self.analyses.update(new)

    def set_configuration_from_file(self, configuration_file):
        try:
            cwd = os.path.abspath(os.path.dirname(__file__))
            config_path = os.path.join("config/", configuration_file)
            self.configuration = ckg_utils.get_configuration(os.path.join(cwd, config_path))
        except Exception as err:
            logger.error("Error: {} reading configuration file: {}.".format(err, config_path))

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
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))
        
        return data
    
    def extract_configuration(self, configuration):
        data_name = None
        analysis_types = None
        plot_types = None
        args = None
        if "data" in configuration:
            data_name = configuration["data"]
        if "analyses" in configuration:
            analysis_types = configuration["analyses"]
        if "plots" in configuration:
            plot_types = configuration["plots"]
        if "args" in configuration:
            args = configuration["args"]

        return data_name, analysis_types, plot_types, args
            
    def generate_report(self):
        report = rp.Report(self.dataset_type.capitalize())
        for section in self.configuration:
            for subsection in self.configuration[section]:
                data_name, analysis_types, plot_types, args = self.extract_configuration(self.configuration[section][subsection])
                if data_name in self.data:
                    data = self.data[data_name]
                    result = None 
                    if len(analysis_types) >= 1:
                        for analysis_type in analysis_types:
                            result = ar.AnalysisResult(self.identifier, analysis_type, args, data)
                            self.update_analyses(result.result)
                            if subsection == "regulation":
                                reg_data = result.result[analysis_type]
                                if not reg_data.empty:
                                    sig_hits = list(set(reg_data.loc[reg_data.rejected,"identifier"]))
                                    #sig_names = list(set(reg_data.loc[reg_data.rejected,"name"]))
                                    sig_data = data[sig_hits]
                                    sig_data.index = data['group'].tolist()
                                    self.update_data({"regulated":sig_data})
                            for plot_type in plot_types:
                                plots = result.get_plot(plot_type, subsection+"_"+analysis_type+"_"+plot_type, analysis_type.capitalize())
                                report.update_plots({(analysis_type, plot_type):plots})
                    else:
                        if result is None:
                            dictresult = {}
                            dictresult["_".join(subsection.split(' '))] = data
                            result = ar.AnalysisResult(self.identifier,"_".join(subsection.split(' ')), {}, data, result = dictresult)
                            self.update_analyses(result.result)
                        for plot_type in plot_types:
                            plots = result.get_plot(plot_type, "_".join(subsection.split(' '))+"_"+plot_type, subsection.capitalize())
                            report.update_plots({("_".join(subsection.split(' ')), plot_type): plots})
        return report

class ProteomicsDataset(Dataset):
    def __init__(self, identifier, data={}, analyses={}):
        config_file = "proteomics.yml"
        Dataset.__init__(self, identifier, "proteomics", data=data, analyses=analyses)
        self.set_configuration_from_file(config_file)
        if len(data) == 0:
            self._data = self.query_data()
        
        self.preprocess_dataset()
        
    def preprocess_dataset(self):
        processed_data = self.preprocessing()
        self.update_data({"preprocessed":processed_data})
    
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
