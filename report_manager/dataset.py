import os
import sys
import pandas as pd
import h5py
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
    def __init__(self, identifier, dataset_type, configuration=None, data={}, analyses={}, analysis_queries={}, report=None):
        self._identifier = identifier
        self._dataset_type = dataset_type
        self._configuration = configuration
        self._data = data
        self._analyses = analyses
        self._analysis_queries = analysis_queries
        self._report = report

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

    @property
    def analysis_queries(self):
        return self._analysis_queries

    @analysis_queries.setter
    def analysis_queries(self, analysis_queries):
        self._analysis_queries = analysis_queries

    @property
    def report(self):
        return self._report

    @report.setter
    def report(self, report):
        self._report = report

    def update_report(self, report):
        self._report.update(report)

    def update_analysis_queries(self, query):
        self.analysis_queries.update(query)

    def get_dataset(self, dataset_name):
        if dataset_name in self.data:
            return self.data[dataset_name]
        return None

    def get_datasets(self, dataset_names):
        datasets = {}
        for dataset_name in dataset_names:
            if dataset_name in self.data:
                datasets[dataset_name] = self.data[dataset_name]
        
        return datasets

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

    def get_dataset_data_directory(self, directory="../../data/reports"):
        ckg_utils.checkDirectory(directory)
        id_directory = os.path.join(directory, self.identifier)
        ckg_utils.checkDirectory(id_directory)
        dataset_directory = os.path.join(id_directory, self.dataset_type)
        ckg_utils.checkDirectory(dataset_directory)

        return dataset_directory

    def query_data(self):
        data = {}
        replace = [("PROJECTID", self.identifier)]
        try:
            cwd = os.path.abspath(os.path.dirname(__file__))
            queries_path = "queries/datasets_cypher.yml"
            datasets_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
            if "replace" in self.configuration:
                replace = self.configuration["replace"]
            for query_name in datasets_cypher[self.dataset_type]:
                title = query_name.lower().replace('_',' ')
                query_type = datasets_cypher[self.dataset_type][query_name]['query_type']
                query = datasets_cypher[self.dataset_type][query_name]['query']
                if query_type == "pre":
                    for r,by in replace:
                        query = query.replace(r,by)
                    data[title] = self.send_query(query)
                else:
                    self.update_analysis_queries({query_name.lower(): query})
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))

        return data

    def send_query(self, query):
        driver = connector.getGraphDatabaseConnectionConfiguration()
        data = connector.getCursorData(driver, query)

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
        self.report = rp.Report(identifier=self.dataset_type.capitalize(), plots={})
        for section in self.configuration:
            if section == "args":
                continue
            for subsection in self.configuration[section]:
                data_names, analysis_types, plot_types, args = self.extract_configuration(self.configuration[section][subsection])
                if isinstance(data_names, list):
                    data = self.get_datasets(data_names)
                else:
                    data = self.get_dataset(data_names)
                
                if data is not None and len(data) > 0:
                    if subsection in self.analysis_queries:
                        query = self.analysis_queries[subsection]
                        if "use" in args:
                            for r_id in args["use"]:
                                if r_id == "columns":
                                    rep = ",".join(['"{}"'.format(i) for i in data.columns.tolist()])
                                elif r_id == "index":
                                    rep = ",".join(['"{}"'.format(i) for i in data.index.tolist()])
                                elif r_id in data.columns:
                                    rep = ",".join(['"{}"'.format(i) for i in data[r_id].tolist()])
                                query = query.replace(args["use"][r_id].upper(),rep)
                            data = self.send_query(query)
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
                                    sig_data["sample"] = data["sample"].tolist()
                                    self.update_data({"regulated":sig_data, "regulation_table":reg_data})
                            for plot_type in plot_types:
                                plots = result.get_plot(plot_type, subsection+"_"+analysis_type+"_"+plot_type)
                                self.report.update_plots({(analysis_type, plot_type):plots})
                    else:
                        if result is None:
                            dictresult = {}
                            dictresult["_".join(subsection.split(' '))] = data
                            result = ar.AnalysisResult(self.identifier,"_".join(subsection.split(' ')), args, data, result = dictresult)
                            self.update_analyses(result.result)
                        for plot_type in plot_types:
                            plots = result.get_plot(plot_type, "_".join(subsection.split(' '))+"_"+plot_type)
                            self.report.update_plots({("_".join(subsection.split(' ')), plot_type): plots})

        self.save_dataset()
        self.save_dataset_report()

    def save_dataset(self):
        dataset_directory = self.get_dataset_data_directory()
        store = pd.HDFStore(os.path.join(dataset_directory, self.dataset_type+".h5"))
        for data in self.data:
            name = data.replace(" ", "-")
            store[name] = self.data[data]

        store.close()

    def save_dataset_report(self):
        dataset_directory = self.get_dataset_data_directory()
        self.report.save_report(directory=dataset_directory)

    def load_dataset(self):
        data = {}
        dataset_directory = self.get_dataset_data_directory()
        dataset_store = os.path.join(dataset_directory, self.dataset_type+".h5")
        if os.path.isfile(dataset_store):
            f = h5py.File(filename, 'r')
            for key in list(f.keys()):
                data[key] = pd.read_hdf(filename, key)

        return data

    def load_dataset_report(self):
        dataset_directory = self.get_dataset_data_directory()
        report = self.report.load_report(directory=dataset_directory)
        self.update_report(report)

class MultiOmicsDataset(dataset):
    def __init__(self, identifier, data, analyses={}, report=None):
        config_file = "multiomics.yml"
        Dataset.__init__(self, identifier, "multiomics", data=data, analyses=analyses, analysis_queries={}, report=report)
        self.configuration_from_file(config_file)

    def get_datasets(self, datasets):
        print("This is a multiomics dataset")
        data = {}
        for dataset_type in datasets:
            dataset_name = datasets[dataset_type]
            if dataset_type in self.data:
                if dataset_name in self.data[dataset_type]:
                    data[dataset_name] = self.data[dataset_type][dataset_name]
        
        return data

class ProteomicsDataset(Dataset):
    def __init__(self, identifier, data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "proteomics.yml"
        Dataset.__init__(self, identifier, "proteomics", data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
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
            index=['group', 'sample', 'subject']
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

            processed_data = basicAnalysis.get_proteomics_measurements_ready(data, index=index, imputation = imputation, method = method, missing_method = missing_method, missing_max = missing_max)
        return processed_data

class LongitudinalProteomicsDataset(ProteomicsDataset):
    def __init__(self, identifier, data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "longitudinal_proteomics.yml"
        ProteomicsDataset.__init__(self, identifier, data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
        self.set_configuration_from_file(config_file)

class ClinicalDataset(Dataset):
    def __init__(self, identifier, data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "clinical.yml"
        Dataset.__init__(self, identifier, "clinical", data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
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
            index = 'index'
            columns = 'clinical_variable'
            values = 'value'
            extra = []
            use_index = False
            args = {}
            if "args" in self.configuration:
                args = self.configuration["args"]
                if "index" in args:
                    index = args["index"]
                if "columns" in args:
                    columns = args["columns"]
                if "values" in args:
                    values = args["values"]
                if "extra" in args:
                    extra = args["extra"]
                if "use_index" in args:
                    use_index = args["use_index"]

            processed_data = basicAnalysis.transform_into_long_format(data, index=index, columns=columns, values=values, extra=extra, use_index=use_index)
        return processed_data

class DNAseqDataset(Dataset):
    def __init__(self, identifier, dataset_type, data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "DNAseq.yml"
        Dataset.__init__(self, identifier, dataset_type=dataset_type, data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
        self.set_configuration_from_file(config_file)
        if len(data) == 0:
            self._data = self.query_data()

class RNAseqDataset(Dataset):
    def __init__(self, identifier, data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "RNAseq.yml"
        Dataset.__init__(self, identifier, "RNAseq", data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
        self.set_configuration_from_file(config_file)
        if len(data) == 0:
            self._data = self.query_data()
