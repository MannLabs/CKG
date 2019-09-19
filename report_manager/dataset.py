import time
import os
import sys
import json
import pandas as pd
import h5py as h5
import ckg_utils
import config.ckg_config as ckg_config
from report_manager import analysisResult as ar, report as rp, utils, knowledge
from report_manager.analyses import basicAnalysis
from report_manager.plots import basicFigures
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

    def generate_dataset(self):
        pass
        
    def update_report(self, report):
        self._report.update(report)

    def update_analysis_queries(self, query):
        self.analysis_queries.update(query)

    def get_dataframe(self, dataset_name):
        if dataset_name in self.data:
            return self.data[dataset_name]
        return None

    def get_dataframes(self, dataset_names):
        datasets = {}
        for dataset_name in dataset_names:
            if dataset_name in self.data:
                datasets[dataset_name] = self.data[dataset_name]

        return datasets

    def list_dataframes(self):
        return list(self.data.keys())

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
    
    def update_configuration_from_file(self, configuration_file):
        try:
            cwd = os.path.abspath(os.path.dirname(__file__))
            config_path = os.path.join("config/", configuration_file)
            self.configuration.update(ckg_utils.get_configuration(os.path.join(cwd, config_path)))
        except Exception as err:
            logger.error("Error: {} reading configuration file: {}.".format(err, config_path))
        

    def get_dataset_data_directory(self, directory="../../../data/reports"):
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
                for r,by in replace:
                    query = query.replace(r,by)
                if query_type == "pre":
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
        description = None
        analysis_types = []
        plot_types = []
        store_analysis = False
        args = None
        if "description" in configuration:
            description = configuration["description"]
        if "data" in configuration:
            data_name = configuration["data"]
        if "analyses" in configuration:
            analysis_types = configuration["analyses"]
        if "plots" in configuration:
            plot_types = configuration["plots"]
        if "store_analysis" in configuration:
            store_analysis = configuration["store_analysis"]
        if "args" in configuration:
            args = configuration["args"]
        
        return description, data_name, analysis_types, plot_types, store_analysis, args

    def add_configuration_to_report(self, report_pipeline):
        conf_plot = basicFigures.generate_configuration_tree(report_pipeline, self.dataset_type)
        self.report.update_plots({('0', self.dataset_type+'_pipeline', 'cytoscape_network'):[conf_plot]})

    def generate_report(self):
        self.report = rp.Report(identifier=self.dataset_type.capitalize(), plots={})
        order = 1
        report_pipeline = {}
        for section in self.configuration:
            report_step = {}
            report_step[section] = {}
            if section == "args":
                continue
            for subsection in self.configuration[section]:
                description, data_names, analysis_types, plot_types, store_analysis, args = self.extract_configuration(self.configuration[section][subsection])
                if description is not None:
                    description = basicFigures.get_markdown(description, args={})
                report_step[section][subsection] = {'data' : data_names, 'analyses': [], 'args': {}}
                if isinstance(data_names, dict) or isinstance(data_names, list):
                    data = self.get_dataframes(data_names)
                else:
                    data = self.get_dataframe(data_names)

                if data is not None and len(data) > 0:
                    if subsection in self.analysis_queries:
                        query = self.analysis_queries[subsection]
                        if "use" in args:
                            for r_id in args["use"]:
                                if r_id == "columns":
                                    rep_str = args["use"][r_id].upper()
                                    rep = ",".join(['"{}"'.format(i) for i in data.columns.unique().tolist()])
                                elif r_id == "index":
                                    rep_str = args["use"][r_id].upper()
                                    rep = ",".join(['"{}"'.format(i) for i in data.index.unique().tolist()])
                                elif r_id in data.columns:
                                    rep_str = r_id.upper()
                                    rep = ",".join(['"{}"'.format(i) for i in data[r_id].unique().tolist()])
                                query = query.replace(rep_str,rep)
                            data = self.send_query(query)
                    result = None
                    if description is not None:
                        self.report.update_plots({(str(order), subsection+"_description", 'description'):[description]})
                        order +=1
                    if len(analysis_types) >= 1:
                        for analysis_type in analysis_types:
                            result = ar.AnalysisResult(self.identifier, analysis_type, args, data)
                            analysis_type = result.analysis_type
                            if analysis_type in result.result and result.result[analysis_type] is not None and len(result.result[analysis_type]) >=1:
                                report_step[section][subsection]['analyses'].append(analysis_type)
                                report_step[section][subsection]['args'] = result.args
                                report_pipeline.update(report_step)
                                self.update_analyses(result.result)
                                if store_analysis:
                                    if analysis_type.lower() == "anova" or analysis_type.lower() == "samr" or analysis_type.lower() == "ttest":
                                        reg_data = result.result[analysis_type]
                                        if not reg_data.empty:
                                            sig_hits = list(set(reg_data.loc[reg_data.rejected,"identifier"]))
                                            sig_data = data[sig_hits]
                                            self.update_data({"regulated":sig_data, "regulation table":reg_data})
                                    else:
                                        self.update_data({subsection+"_"+analysis_type: result.result[analysis_type]})
                                for plot_type in plot_types:
                                    plots = result.get_plot(plot_type, subsection+"_"+analysis_type+"_"+plot_type)
                                    self.report.update_plots({(str(order), subsection+"_"+analysis_type, plot_type):plots})
                                    order +=1
                    else:
                        if result is None:
                            dictresult = {}
                            dictresult["_".join(subsection.split(' '))] = data
                            result = ar.AnalysisResult(self.identifier,"_".join(subsection.split(' ')), args, data, result = dictresult)
                            report_pipeline.update(report_step)
                            self.update_analyses(result.result)
                        for plot_type in plot_types:
                            plots = result.get_plot(plot_type, "_".join(subsection.split(' '))+"_"+plot_type)
                            self.report.update_plots({(str(order), "_".join(subsection.split(' ')), plot_type): plots})
                            order += 1
        
        self.add_configuration_to_report(report_pipeline)


    def save_dataset_recursively(self, dset, group, dt):
        max_size = 20
        df_set = None
        for name in dset:
            if isinstance(dset[name], dict):
                grp = group.create_group(name)
                df_set = self.save_dataset_recursively(dset[name], grp, dt)
            elif isinstance(dset[name], pd.DataFrame):
                if dset[name].memory_usage().sum()/1000000 < max_size: #Only store if memory usage below 20Mb
                    if not dset[name].index.is_numeric():
                        dset[name] = dset[name].reset_index()
                    df_set = group.create_dataset(name, (1,), dtype=dt, compression="gzip", chunks=True, data=dset[name].to_json(orient='records'))
        
        return df_set
    
    def save_dataset(self, dataset_directory):
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_directory)
        if len(self.data) > 0:
            dt = h5.special_dtype(vlen=str) 
            with h5.File(os.path.join(dataset_directory, self.dataset_type+"_dataset.h5"), "w") as f:
                grp = f.create_group(self.dataset_type)
                df_set = self.save_dataset_recursively(self.data, grp, dt)

    def save_dataset_recursively_to_file(self, dset, dataset_directory, base_name=''):
        for name in dset:
            if isinstance(dset[name], dict):
                self.save_dataset_recursively_to_file(dset[name], dataset_directory, name+"_"+base_name)
            elif isinstance(dset[name], pd.DataFrame):
                dset[name].to_csv(os.path.join(dataset_directory,name), sep='\t', header=True, index=False, quotechar='"', line_terminator='\n', escapechar='\\')
                        
    def save_dataset_to_file(self, dataset_directory):
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_directory)
        self.save_dataset_recursively_to_file(self.data, dataset_directory, base_name='')
        ckg_utils.save_dict_to_yaml(self.configuration, os.path.join(dataset_directory, self.dataset_type+".yml"))
        
    def save_report(self, dataset_directory):
        if not os.path.exists(dataset_directory):
                os.makedirs(dataset_directory)
        self.report.save_report(directory=dataset_directory)
        
    def load_dataset_recursively(self, dset, loaded_dset={}):
        for name in dset:
            if isinstance(dset[name], h5._hl.group.Group):
                loaded_dset[name] = {}
                loaded_dset[name] = self.load_dataset_recursively(dset[name], loaded_dset[name])
            else:
                loaded_dset[name] = pd.read_json(dset[name][0], orient='records')
        
        return loaded_dset

    def load_dataset(self, dataset_directory):
        dataset_store = os.path.join(dataset_directory, self.dataset_type+"_dataset.h5")
        if os.path.isfile(dataset_store):
            with h5.File(dataset_store, 'r') as f:
                if self.dataset_type in f:
                    self.data.update(self.load_dataset_recursively(f[self.dataset_type], {}))
                            
    def load_dataset_report(self, report_dir):
        self.load_dataset(report_dir)
        dataset_dir = os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)),report_dir), self.dataset_type)
        r = rp.Report(self.dataset_type,{})
        r.read_report(dataset_dir)
        self.report = r
        
    def generate_knowledge(self):
        kn = knowledge.Knowledge(self.identifier, self.data, nodes={}, relationships={}, queries_file=None, colors={}, graph=None, report={})
        
        return kn

class MultiOmicsDataset(Dataset):
    def __init__(self, identifier, data, analyses={}, analysis_queries={}, report=None):
        self._config_file = "multiomics.yml"
        Dataset.__init__(self, identifier, "multiomics", data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
        self.set_configuration_from_file(self._config_file)

    def get_dataframes(self, datasets):
        data = {}
        for dataset_type in datasets:
            dataset_name = datasets[dataset_type]
            if dataset_type in self.data:
                dataset = self.data[dataset_type]
                data[dataset_type] = self.data[dataset_type].get_dataframe(dataset_name)

        return data
    
    def generate_knowledge(self):
        kn = knowledge.MultiOmicsKnowledge(self.identifier, self.data, nodes={}, relationships={}, colors={}, graph=None, report={})
        kn.generate_knowledge()        
        
        return kn
    
class ProteomicsDataset(Dataset):
    def __init__(self, identifier, dataset_type="proteomics", data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "proteomics.yml"
        Dataset.__init__(self, identifier, dataset_type, data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
        self.set_configuration_from_file(config_file)

    def generate_dataset(self):
        self._data = self.query_data()
        self.process_dataset()

    def process_dataset(self):
        processed_data = self.processing()
        self.update_data({"processed":processed_data})

    def processing(self):
        processed_data = None
        data = self.get_dataframe("original")
        if data is not None:
            imputation = True
            method = "mixed"
            missing_method = 'percentage'
            missing_max = 0.3
            min_valid = 1
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
                if "min_valid" in args:
                    min_valid = args['min_valid']
                if "value_col" in args:
                    value_col = args["value_col"]

            processed_data = basicAnalysis.get_proteomics_measurements_ready(data, index_cols=index, imputation = imputation, method = method, missing_method = missing_method, missing_max = missing_max, min_valid=min_valid)
        return processed_data
    
    def generate_knowledge(self):
        kn = knowledge.ProteomicsKnowledge(self.identifier, self.data, nodes={}, relationships={}, colors={}, graph=None, report={})
        kn.generate_knowledge()        
        
        return kn

class LongitudinalProteomicsDataset(ProteomicsDataset):
    def __init__(self, identifier, data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "longitudinal_proteomics.yml"
        ProteomicsDataset.__init__(self, identifier, data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
        #self.dataset_type = "longitudinal_proteomics"
        self.update_configuration_from_file(config_file)

class ClinicalDataset(Dataset):
    def __init__(self, identifier, data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "clinical.yml"
        Dataset.__init__(self, identifier, "clinical", data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
        self.set_configuration_from_file(config_file)

    def generate_dataset(self):
        self._data = self.query_data()
        self.process_dataset()

    def process_dataset(self):
        processed_data = self.processing()
        self.update_data({"processed":processed_data})

    def processing(self):
        processed_data = None
        data = self.get_dataframe("original")
        if data is not None:
            subject_id = 'subject'
            sample_id = 'biological_sample'
            group_id = 'group'
            columns = 'clinical_variable'
            values = 'values'
            extra = []
            imputation  = True
            imputation_method = 'KNN'
            args = {}
            if "args" in self.configuration:
                args = self.configuration["args"]
                if "subject_id" in args:
                    subject_id = args["subject_id"]
                if "sample_id" in args:
                    sample_id = args["sample_id"]
                if "columns" in args:
                    columns = args["columns"]
                if "values" in args:
                    values = args["values"]
                if "extra" in args:
                    extra = args["extra"]
                if 'imputation_method' in args:
                    imputation = True
                    imputation_method = args['imputation_method']
                if 'group_id' in args:
                    group_id = args['group_id']

            processed_data = basicAnalysis.get_clinical_measurements_ready(data, subject_id=subject_id, sample_id=sample_id, group_id=group_id, columns=columns, values=values, extra=extra, imputation=imputation, imputation_method=imputation_method)
        return processed_data
    
    def generate_knowledge(self):
        kn = knowledge.ClinicalKnowledge(self.identifier, self.data, nodes={}, relationships={}, colors={}, graph=None, report={})
        kn.generate_knowledge()        
        
        return kn

class DNAseqDataset(Dataset):
    def __init__(self, identifier, dataset_type, data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "DNAseq.yml"
        Dataset.__init__(self, identifier, dataset_type=dataset_type, data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
        self.set_configuration_from_file(config_file)

    def generate_dataset(self):
        self._data = self.query_data()

class RNAseqDataset(Dataset):
    def __init__(self, identifier, data={}, analyses={}, analysis_queries={}, report=None):
        config_file = "RNAseq.yml"
        Dataset.__init__(self, identifier, "RNAseq", data=data, analyses=analyses, analysis_queries=analysis_queries, report=report)
        self.set_configuration_from_file(config_file)

    def generate_dataset(self):
        self._data = self.query_data()
