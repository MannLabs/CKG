import os
import sys
import pandas as pd
import h5py as h5
from collections import OrderedDict
from ckg import ckg_utils
from ckg.report_manager import report as rp, knowledge
from ckg.analytics_core import analytics_factory
from ckg.analytics_core.analytics import analytics
from ckg.analytics_core.viz import viz
from ckg.graphdb_connector import connector

ckg_config = ckg_utils.read_ckg_config()
cwd = os.path.dirname(os.path.abspath(__file__))
log_config = ckg_config['report_manager_log']
logger = ckg_utils.setup_logging(log_config, key="dataset")


class Dataset:
    def __init__(self, identifier, dataset_type, configuration=None, data={}, analysis_queries={}, report=None):
        self._identifier = identifier
        self._dataset_type = dataset_type
        self._configuration = configuration
        self._data = data
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

    def update_data(self, new):
        self.data.update(new)

    def set_configuration_from_file(self, configuration_file):
        try:
            config_path = os.path.join("config/", configuration_file)
            self.configuration = ckg_utils.get_configuration(os.path.join(cwd, config_path))
        except Exception as err:
            logger.error("Error: {} reading configuration file: {}.".format(err, config_path))

    def update_configuration_from_file(self, configuration_file):
        try:
            config_path = os.path.join("config/", configuration_file)
            if os.path.exists(os.path.join(cwd, config_path)):
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
            queries_path = "queries/datasets_cypher.yml"
            datasets_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
            if "replace" in self.configuration:
                replace = self.configuration["replace"]
            for query_name in datasets_cypher[self.dataset_type]:
                title = query_name.lower().replace('_', ' ')
                query_type = datasets_cypher[self.dataset_type][query_name]['query_type']
                query = datasets_cypher[self.dataset_type][query_name]['query']
                for r, by in replace:
                    query = query.replace(r, by)
                if query_type == "pre":
                    data[title] = self.send_query(query)
                else:
                    self.update_analysis_queries({query_name.lower(): query})
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Reading queries from file {}: {}, file: {},line: {}, err: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno, err))

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
        conf_plot = viz.generate_configuration_tree(report_pipeline, self.dataset_type)
        self.report.update_plots({('0', self.dataset_type+'_pipeline', 'cytoscape_network'): [conf_plot]})

    def generate_report(self):
        self.report = rp.Report(identifier=self.dataset_type.capitalize(), plots={})
        order = 1
        report_pipeline = OrderedDict()
        if self.configuration is not None:
            for section in self.configuration:
                report_step = {}
                report_step[section] = {}
                if section == "args":
                    continue
                for subsection in self.configuration[section]:
                    description, data_names, analysis_types, plot_types, store_analysis, args = self.extract_configuration(self.configuration[section][subsection])
                    if description is not None:
                        description = viz.get_markdown(description, args={})
                    report_step[section][subsection] = {'data': data_names, 'analyses': [], 'args': {}}
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
                                    query = query.replace(rep_str, rep)
                                data = self.send_query(query)
                        result = None
                        if description is not None:
                            self.report.update_plots({(str(order), subsection+"_description", 'description'): [description]})
                            order += 1
                        if len(analysis_types) >= 1:
                            for analysis_type in analysis_types:
                                result = analytics_factory.Analysis(subsection, analysis_type, args, data)
                                analysis_type = result.analysis_type
                                if analysis_type in result.result and result.result[analysis_type] is not None and len(result.result[analysis_type]) >=1:
                                    report_step[section][subsection]['analyses'].append(analysis_type)
                                    report_step[section][subsection]['args'] = result.args
                                    report_pipeline.update(report_step)
                                    if store_analysis:
                                        if analysis_type.lower() in ["anova", "samr", "ttest", "ancova", "mixed_anova"]:
                                            reg_data = result.result[analysis_type]
                                            if not reg_data.empty:
                                                if isinstance(data, dict):
                                                    data = data['processed']
                                                cols = []
                                                if 'group' in data.columns:
                                                    cols.append('group')
                                                if 'sample' in data.columns:
                                                    cols.append('sample')
                                                if 'subject' in data.columns:
                                                    cols.append('subject')
                                                if 'within' in data.columns:
                                                    cols.append('within')
                                                if 'between' in data.columns:
                                                    cols.append('between')
                                                sig_hits = list(set(reg_data.loc[reg_data.rejected,"identifier"])) + cols
                                                sig_data = data[sig_hits]
                                                self.update_data({"regulated": sig_data, "regulation table": reg_data})
                                        else:
                                            self.update_data({subsection + "_" + analysis_type: result.result[analysis_type]})
                                    for plot_type in plot_types:
                                        plots = result.get_plot(plot_type, subsection + "_" + analysis_type + "_" + plot_type)
                                        self.report.update_plots({(str(order), subsection + "_" + analysis_type, plot_type): plots})
                                        order += 1
                        else:
                            if result is None:
                                dictresult = {}
                                dictresult["_".join(subsection.split(' '))] = data
                                result = analytics_factory.Analysis(self.identifier, "_".join(subsection.split(' ')), args, data, result=dictresult)
                                report_pipeline.update(report_step)
                                if store_analysis:
                                    self.update_data({"_".join(subsection.split(' ')): data})
                            for plot_type in plot_types:
                                plots = result.get_plot(plot_type, "_".join(subsection.split(' '))+"_"+plot_type)
                                self.report.update_plots({(str(order), "_".join(subsection.split(' ')), plot_type): plots})
                                order += 1

        self.add_configuration_to_report(report_pipeline)

    def save_dataset_recursively(self, dset, group, dt):
        df_set = None
        for name in dset:
            if isinstance(dset[name], dict):
                grp = group.create_group(name)
                df_set = self.save_dataset_recursively(dset[name], grp, dt)
            elif isinstance(dset[name], pd.DataFrame):
                if not dset[name].index.is_numeric():
                    dset[name] = dset[name].reset_index()
                try:
                    df_set = group.create_dataset(name, (1,), dtype=dt, compression="gzip", chunks=True, data=dset[name].to_json(orient='records'))
                except ValueError:
                    print("Could not save dataset: {}. Memory usage {}".format(name, dset[name].memory_usage().sum()/1000000))
                    print(dset[name])

        return df_set

    def save_dataset(self, dataset_directory):
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_directory)
        if len(self.data) > 0:
            dt = h5.special_dtype(vlen=str)
            with h5.File(os.path.join(dataset_directory, self.dataset_type + "_dataset.h5"), "w") as f:
                grp = f.create_group(self.dataset_type)
                df_set = self.save_dataset_recursively(self.data, grp, dt)

    def save_dataset_recursively_to_file(self, dset, dataset_directory, base_name=''):
        for name in dset:
            if isinstance(dset[name], dict):
                self.save_dataset_recursively_to_file(dset[name], dataset_directory, name+"_"+base_name)
            elif isinstance(dset[name], pd.DataFrame):
                filename = name+".tsv"
                if base_name != '':
                    filename = name+"_"+base_name+".tsv"
                dset[name].to_csv(os.path.join(dataset_directory, filename), sep='\t', header=True, index=False, quotechar='"', line_terminator='\n', escapechar='\\')

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
        dataset_dir = os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), report_dir), self.dataset_type)
        r = rp.Report(self.dataset_type, {})
        r.read_report(dataset_dir)
        self.report = r

    def generate_knowledge(self):
        kn = knowledge.Knowledge(self.identifier, self.data, nodes={}, relationships={}, queries_file=None, colors={}, graph=None, report={})

        return kn


class MultiOmicsDataset(Dataset):
    def __init__(self, identifier, data, configuration=None, analysis_queries={}, report=None):
        Dataset.__init__(self, identifier, "multiomics", data=data, configuration=configuration, analysis_queries=analysis_queries, report=report)
        if configuration is None:
            self._config_file = "multiomics.yml"
            self.set_configuration_from_file(self._config_file)

    def get_dataframes(self, datasets):
        data = {}
        for dataset_type in datasets:
            if isinstance(datasets, dict):
                dataset_name = datasets[dataset_type]
            else:
                dataset_name = dataset_type
            if dataset_type in self.data:
                dataset = self.data[dataset_type]
                data[dataset_type] = dataset.get_dataframe(dataset_name)

        return data

    def generate_knowledge(self, nodes):
        kn = knowledge.MultiOmicsKnowledge(self.identifier, self.data, nodes=nodes, relationships={}, colors={}, graph=None, report={})
        kn.generate_knowledge()

        return kn


class ProteomicsDataset(Dataset):
    def __init__(self, identifier, dataset_type="proteomics", data={}, configuration=None, analysis_queries={}, report=None):
        Dataset.__init__(self, identifier, dataset_type, data=data, configuration=configuration, analysis_queries=analysis_queries, report=report)
        if configuration is None:
            config_file = "proteomics.yml"
            self.set_configuration_from_file(config_file)

    def generate_dataset(self):
        self._data = self.query_data()
        self.process_dataset()
        if "args" in self.configuration:
            args = self.configuration["args"]
            if "batch_correction" in args and args["batch_correction"]:
                    self.correct_batches()
            else:
                data = self.get_dataframe("processed")
                batch_col = 'batch'
                if 'batch_col' in args:
                    batch_col = args['batch_col']

                if batch_col in data:
                    data = data.drop(batch_col, axis=1)
                self.update_data({"processed": data})

        self.widen_metadata_dataset()

    def process_dataset(self):
        processed_data = self.processing()
        self.update_data({"processed": processed_data})
        
    def correct_batches(self):
        data = self.get_dataframe("processed")
        corrected_data = None
        if data is not None and not data.empty:
            if "args" in self.configuration:
                args = self.configuration["args"]
                batch_col = 'batch'
                if 'batch_col' in args:
                    batch_col = args["batch_col"]
                
                index_cols = ['subject', 'sample', 'group']
                if 'index' in args:
                    index_cols = [c for c in args['index'] if c != batch_col]
                    
                corrected_data = analytics.combat_batch_correction(data, batch_col=batch_col, index_cols=index_cols)

        if not corrected_data.empty:
            self.update_data({"uncorrected_batch": data, "processed": corrected_data})
        else:
            self.update_data({"processed": data.drop(batch_col, axis=1)})

    def widen_metadata_dataset(self):
        data = self.get_dataframe("metadata")
        wdata = analytics.transform_into_wide_format(data, index=['subject', 'biological_sample'], columns='clinical_variable', values='value', extra=['group', 'group2'])
        self.update_data({"metadata": wdata})

    def processing(self):
        processed_data = None
        data = self.get_dataframe("original")

        if data is not None:
            if not data.empty:
                filter_samples = False
                filter_samples_percent = 0.5
                imputation = True
                method = "mixed"
                missing_method = 'percentage'
                missing_per_group = True
                missing_max = 0.3
                min_valid = 1
                value_col = 'LFQ_intensity'
                index = ['group', 'sample', 'subject', 'batch']
                extra_identifier = None
                shift = 1.8
                nstd = 0.3
                knn_cutoff = 0.6
                normalize = False
                normalization_method = 'median'
                normalize_group = False
                normalize_by = None
                args = {}
                if "args" in self.configuration:
                    args = self.configuration["args"]
                    if "filter_samples" in args:
                        filter_samples = args['filter_samples']
                    if "filter_samples_percent" in args:
                        filter_samples_percent = args['filter_samples_percent']
                    if "imputation" in args:
                        imputation = args["imputation"]
                    if "extra_identifier" in args:
                        extra_identifier = args["extra_identifier"]
                    if "imputation_method" in args:
                        method = args["imputation_method"]
                    if "missing_method" in args:
                        missing_method = args["missing_method"]
                    if "missing_per_group" in args:
                        missing_per_group = args["missing_per_group"]
                    if "missing_max" in args:
                        missing_max = args["missing_max"]
                    if "min_valid" in args:
                        min_valid = args['min_valid']
                    if "value_col" in args:
                        value_col = args["value_col"]
                    if "missing_nstd" in args:
                        nstd = args["missing_nstd"]
                    if "missing_shift" in args:
                        shift = args["missing_shift"]
                    if "knn_cutoff" in args:
                        knn_cutoff = args["knn_cutoff"]
                    if "normalize" in args:
                        normalize = args["normalize"]
                    if "normalization_method" in args:
                        normalization_method = args["normalization_method"]
                    if "normalize_group" in args:
                        normalize_group = args["normalize_group"]
                    if "normalize_by" in args:
                        normalize_by = args["normalize_by"]
                    if 'index' in args:
                        index = args['index']

                processed_data = analytics.get_proteomics_measurements_ready(data, index_cols=index, imputation=imputation,
                                                                             imputation_method=method, missing_method=missing_method, extra_identifier=extra_identifier,
                                                                             filter_samples=filter_samples, filter_samples_percent=filter_samples_percent,
                                                                             missing_per_group=missing_per_group, missing_max=missing_max,
                                                                             min_valid=min_valid, shift=shift, nstd=nstd, value_col=value_col, knn_cutoff=knn_cutoff,
                                                                             normalize=normalize, normalization_method=normalization_method, normalize_group=normalize_group, normalize_by=normalize_by)
        return processed_data

    def generate_knowledge(self):
        kn = knowledge.ProteomicsKnowledge(self.identifier, self.data, nodes={}, relationships={}, colors={}, graph=None, report={})
        kn.generate_knowledge()

        return kn


class PTMDataset(ProteomicsDataset):
    def __init__(self, identifier, data={}, configuration=None, analysis_queries={}, report=None):
        ProteomicsDataset.__init__(self, identifier, dataset_type="ptm", data=data, configuration=configuration, analysis_queries=analysis_queries, report=report)


class PhosphoproteomicsDataset(PTMDataset):
    def __init__(self, identifier, data={}, configuration=None, analysis_queries={}, report=None):
        PTMDataset.__init__(self, identifier, data=data, configuration=configuration, analysis_queries=analysis_queries, report=report)
        if configuration is None:
            config_file = "phosphoproteomics.yml"
            self.update_configuration_from_file(config_file)


class InteractomicsDataset(ProteomicsDataset):
    def __init__(self, identifier, data={}, configuration=None, analysis_queries={}, report=None):
        ProteomicsDataset.__init__(self, identifier, data=data, configuration=configuration, analysis_queries=analysis_queries, report=report)
        if configuration is None:
            config_file = "interactomics.yml"
            self.update_configuration_from_file(config_file)


class LongitudinalProteomicsDataset(ProteomicsDataset):
    def __init__(self, identifier, data={}, configuration=None, analysis_queries={}, report=None):
        ProteomicsDataset.__init__(self, identifier, data=data, configuration=configuration, analysis_queries=analysis_queries, report=report)
        if configuration is None:
            config_file = "longitudinal_proteomics.yml"
            self.update_configuration_from_file(config_file)


class ClinicalDataset(Dataset):
    def __init__(self, identifier, data={}, configuration=None, analysis_queries={}, report=None):
        Dataset.__init__(self, identifier, "clinical", data=data, configuration=configuration, analysis_queries=analysis_queries, report=report)
        if configuration is None:
            config_file = "clinical.yml"
            self.set_configuration_from_file(config_file)

    def generate_dataset(self):
        self._data = self.query_data()
        self.process_dataset()
        self.widen_original_dataset()

    def process_dataset(self):
        processed_data = self.processing()
        self.update_data({"processed": processed_data})

    def widen_original_dataset(self):
        data = self.get_dataframe("original")
        wdata = analytics.transform_into_wide_format(data, index=['subject', 'biological_sample'], columns='clinical_variable', values='value', extra=['group', 'group2'])
        self.update_data({"original": wdata})

    def processing(self):
        processed_data = None
        data = self.get_dataframe("original")
        if data is not None:
            if not data.empty:
                subject_id = 'subject'
                sample_id = 'biological_sample'
                group_id = 'group'
                columns = 'clinical_variable'
                values = 'values'
                extra = []
                imputation = True
                imputation_method = 'KNN'
                missing_method = 'percentage'
                missing_max = 0.3
                min_valid = 1
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
                    if "missing_method" in args:
                        missing_method = args["missing_method"]
                    if "missing_max" in args:
                        missing_max = args["missing_max"]
                    if "min_valid" in args:
                        min_valid = args['min_valid']
                    if 'group_id' in args:
                        group_id = args['group_id']

                processed_data = analytics.get_clinical_measurements_ready(data, subject_id=subject_id, sample_id=sample_id,
                                                                           group_id=group_id, columns=columns, values=values,
                                                                           extra=extra, imputation=imputation, imputation_method=imputation_method,
                                                                           missing_method=missing_method, missing_max=missing_max, min_valid=min_valid)

        return processed_data

    def generate_knowledge(self):
        kn = knowledge.ClinicalKnowledge(self.identifier, self.data, nodes={}, relationships={}, colors={}, graph=None, report={})
        kn.generate_knowledge()

        return kn


#ToDO
class DNAseqDataset(Dataset):
    def __init__(self, identifier, dataset_type, data={}, configuration=None, analysis_queries={}, report=None):

        Dataset.__init__(self, identifier, dataset_type=dataset_type, data=data, configuration=configuration, analysis_queries=analysis_queries, report=report)
        if configuration is None:
            config_file = "DNAseq.yml"
            self.set_configuration_from_file(config_file)

    def generate_dataset(self):
        self._data = self.query_data()

#ToDO
class RNAseqDataset(Dataset):
    def __init__(self, identifier, data={}, configuration=None, analysis_queries={}, report=None):

        Dataset.__init__(self, identifier, "RNAseq", data=data, configuration=configuration, analysis_queries=analysis_queries, report=report)
        if configuration is None:
            config_file = "RNAseq.yml"
            self.set_configuration_from_file(config_file)

    def generate_dataset(self):
        self._data = self.query_data()
