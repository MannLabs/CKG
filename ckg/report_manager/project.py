import sys
import os
import time
import h5py as h5
import json
from collections import defaultdict
from json import dumps
import pandas as pd
from ckg import ckg_utils
from ckg.graphdb_builder import builder_utils
from ckg.report_manager.dataset import Dataset, DNAseqDataset, ProteomicsDataset, InteractomicsDataset, PhosphoproteomicsDataset, ClinicalDataset, LongitudinalProteomicsDataset, MultiOmicsDataset
from ckg.analytics_core.viz import viz
from ckg.analytics_core import utils as acore_utils
from ckg.report_manager import report as rp, utils, knowledge
from ckg.graphdb_connector import query_utils
from ckg.graphdb_connector import connector

ckg_config = ckg_utils.read_ckg_config()
cwd = os.path.dirname(os.path.abspath(__file__))
log_config = ckg_config['report_manager_log']
logger = ckg_utils.setup_logging(log_config, key="project")


class Project:
    """
    A project class that defines an experimental project.
    A project can be of different types, contain several datasets and reports.

    Example::

        p = Project(identifier="P0000001", datasets=None, report=None)
        p.show_report(environment="notebook")
    """

    def __init__(self, identifier, configuration_files={}, datasets={}, knowledge=None, report={}):
        self._identifier = identifier
        self._queries_file = 'queries/project_cypher.yml'
        self.configuration_files = configuration_files
        self._datasets = datasets
        self._knowledge = knowledge
        self._report = report
        self._name = None
        self._acronym = None
        self._data_types = []
        self._responsible = None
        self._description = None
        self._status = None
        self._num_subjects = None
        self._similar_projects = None
        self._overlap = None

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        self._identifier = identifier

    @property
    def configuration_files(self):
        return self._configuration_files

    @configuration_files.setter
    def configuration_files(self, configuration_files):
        self._configuration_files = configuration_files

    @property
    def queries_file(self):
        return self._queries_file

    @queries_file.setter
    def queries_file(self, queries_file):
        self._queries_file = queries_file

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

    def append_data_type(self, data_type):
        self._data_types.append(data_type)

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
    def knowledge(self):
        return self._knowledge

    @knowledge.setter
    def knowledge(self, knowledge):
        self._knowledge = knowledge

    @property
    def report(self):
        return self._report

    @report.setter
    def report(self, report):
        self._report = report

    @property
    def similar_projects(self):
        return self._similar_projects

    @similar_projects.setter
    def similar_projects(self, similarity_matrix):
        self._similar_projects = similarity_matrix

    @property
    def overlap(self):
        return self._overlap

    @overlap.setter
    def overlap(self, overlap_matrix):
        self._overlap = overlap_matrix

    def get_dataset(self, dataset):
        if dataset in self.datasets:
            return self.datasets[dataset]
        return None

    def update_dataset(self, dataset):
        self.datasets.update(dataset)

    def update_report(self, new):
        self.report.update(new)
        
    def get_sdrf(self):
        sdrf_df = pd.DataFrame()
        try:
            driver = connector.getGraphDatabaseConnectionConfiguration()
            query_path = os.path.join(cwd, self.queries_file)
            project_cypher = query_utils.read_queries(query_path)
            query = query_utils.get_query(project_cypher, query_id="project_sdrf")
            df = connector.getCursorData(driver, query.replace("PROJECTID", self.identifier))
            sdrf_df = builder_utils.convert_ckg_to_sdrf(df)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Error: {}. Reading queries from file {}: {}, file: {},line: {}".format(err, query_path, sys.exc_info(), fname, exc_tb.tb_lineno))
        
        return sdrf_df

    def remove_project(self, host="localhost", port=7687, user="neo4j", password="password"):
        try:
            query_path = os.path.join(cwd, self.queries_file)
            project_cypher = query_utils.read_queries(query_path)
            query = query_utils.get_query(project_cypher, query_id="remove_project")
            driver = connector.connectToDB(host, port, user, password)
            queries = query.replace("PROJECTID", self.identifier)
            for query in queries.split(';')[:-1]:
                result = connector.sendQuery(driver, query+';', parameters={})
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Error removing project {}. Query file: {},line: {}, error: {}".format(self.identifier, fname, exc_tb.tb_lineno, err))

    def get_report_directory(self):
        reports_dir = os.path.join(ckg_config['reports_directory'], self.identifier)
        if not os.path.isdir(reports_dir):
            os.makedirs(reports_dir)
        return reports_dir

    def get_downloads_directory(self):
        downloads_dir = os.path.join(ckg_config['downloads_directory'], self.identifier)
        if not os.path.isdir(downloads_dir):
            os.makedirs(downloads_dir)
        return downloads_dir

    def set_attributes(self, project_info):
        if "attributes" in project_info:
            attributes = project_info["attributes"].to_dict('r')[0]
            if len(attributes) > 0:
                self.from_dict(attributes)

    def from_dict(self, attributes):
        if "name" in attributes:
            self.name = attributes["name"]
        if "acronym" in attributes:
            self.acronym = attributes["acronym"]
        if "description" in attributes:
            self.description = attributes["description"]
        if "data_types" in attributes:
            if isinstance(attributes['data_types'], str):
                self.data_types = [i.strip(' ') for i in attributes["data_types"].split('|')]
            else:
                self.data_types = attributes["data_types"]
        if "responsible" in attributes:
            if isinstance(attributes['responsible'], str):
                self.responsible = [i.strip(' ') for i in attributes["responsible"].split('|')]
            else:
                self.responsible = attributes['responsible']
        if "status" in attributes:
            self.status = attributes["status"]
        if "number_subjects" in attributes:
            self.num_subjects = attributes["number_subjects"]
        if "similar_projects" in attributes:
            self.similar_projects = pd.DataFrame.from_dict(attributes['similar_projects'])
        if "overlap" in attributes:
            self.overlap = pd.DataFrame.from_dict(attributes['overlap'])

    def to_dict(self):
        similarity_dict = {}
        overlap_dict = {}
        if self.similar_projects is not None:
            similarity_dict = self.similar_projects.to_dict(orient='records')
        if self.overlap is not None:
            overlap_dict = self.overlap.to_dict(orient='records')
        d = {"identifier": self.identifier,
             "queries_file": self._queries_file,
             "name": self.name,
             "acronym": self.acronym,
             "description": self.description,
             "data_types": self.data_types,
             "responsible": self.responsible,
             "status": self.status,
             "number_subjects": self.num_subjects,
             "similar_projects": similarity_dict,
             "overlap": overlap_dict
             }

        return d

    def to_dataframe(self):
        d = self.to_dict()
        df = pd.DataFrame.from_dict(d, orient='index')
        df = df.transpose()

        return df

    def list_datasets(self):
        datasets = None
        if self.datasets is not None:
            datasets = self.datasets.keys()
        return datasets

    def to_json(self):
        d = self.to_dict()
        djson = dumps(d)

        return djson

    def from_json(self, json_str):
        d = json.loads(json_str)
        self.from_dict(d)

    def query_data(self):
        data = {}
        try:
            queries_path = os.path.join(cwd, self.queries_file)
            project_cypher = query_utils.read_queries(queries_path)

            driver = connector.getGraphDatabaseConnectionConfiguration()
            replace = [("PROJECTID", self.identifier)]
            for query_name in project_cypher:
                title = query_name.lower().replace('_', ' ')
                query = project_cypher[query_name]['query']
                query_type = project_cypher[query_name]['query_type']
                for r, by in replace:
                    query = query.replace(r, by)
                if query_type == "pre":
                    data[title] = connector.getCursorData(driver, query)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Reading queries from file {}: {}, file: {},line: {}, error: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno, err))

        return data

    def check_report_exists(self):
        exists = True
        report_dir = os.path.join(ckg_config['reports_directory'], self.identifier)
        if not os.path.isdir(report_dir):
            return False
        for dataset in self.report:
            if os.path.isdir(os.path.join(report_dir, dataset)):
                continue
            exists = False

        return exists

    def load_project_report(self):
        self.load_project_data()
        project_dir = os.path.join(ckg_config['reports_directory'], self.identifier)
        self.report = {}
        for root, data_types, files in os.walk(project_dir):
            for data_type in data_types:
                r = rp.Report(data_type, {})
                r.read_report(os.path.join(root, data_type))
                if data_type in self.datasets:
                    self.datasets[data_type].report = r
                elif data_type == "Knowledge":
                    self.knowledge = knowledge.Knowledge(self.identifier, {'name': self.name}, report=r)
                else:
                    self.update_report({data_type: r})

    def load_project(self, directory):
        dataset_store = os.path.join(directory, "project_information_dataset.h5")
        if os.path.isfile(dataset_store):
            with h5.File(dataset_store, 'r') as f:
                if "Project_information" in f:
                    self.from_json(f["Project_information"][0])

    def load_project_data(self):
        project_dir = os.path.join(ckg_config['reports_directory'], self.identifier)
        self.load_project(os.path.join(project_dir, "Project information"))
        for root, data_types, files in os.walk(project_dir):
            for data_type in data_types:
                dataset = None
                if data_type == "proteomics":
                    dataset = ProteomicsDataset(self.identifier, data={}, analysis_queries={}, report=None)
                elif data_type == "clinical":
                    dataset = ClinicalDataset(self.identifier, data={}, analysis_queries={}, report=None)
                elif data_type == "wes" or data_type == "wgs":
                    dataset = DNAseqDataset(self.identifier, dataset_type=data_type, data={}, analysis_queries={}, report=None)
                elif data_type == "interactomics":
                    dataset = InteractomicsDataset(self.identifier, data={}, analysis_queries={}, report=None)
                elif data_type == "phosphoproteomics":
                    dataset = PhosphoproteomicsDataset(self.identifier, data={}, analysis_queries={}, report=None)
                elif data_type == "longitudinal_proteomics":
                    dataset = LongitudinalProteomicsDataset(self.identifier, data={}, analysis_queries={}, report=None)
                elif data_type == "multiomics":
                    dataset = MultiOmicsDataset(self.identifier, data={}, report=None)

                if dataset is not None:
                    dataset.load_dataset(os.path.join(root, data_type))
                    self.update_dataset({data_type: dataset})

    def build_project(self, force=False):
        if self.check_report_exists() and not force:
            self.load_project_report()
        elif force:
            self.report = {}
            self.datasets = {}

        if len(self.report) == 0 or len(self.datasets) == 0:
            project_info = self.query_data()
            if len(project_info) > 0:
                self.set_attributes(project_info)
                self.get_similar_projects(project_info)
                self.get_projects_overlap(project_info)
                for data_type in self.data_types:
                    dataset = None
                    configuration = None
                    if data_type == "proteomics":
                        if "proteomics" in self.configuration_files:
                            configuration = ckg_utils.get_configuration(self.configuration_files["proteomics"])
                        dataset = ProteomicsDataset(self.identifier, data={}, configuration=configuration, analysis_queries={}, report=None)
                    elif data_type == "clinical":
                        if "clinical" in self.configuration_files:
                            configuration = ckg_utils.get_configuration(self.configuration_files["clinical"])
                        dataset = ClinicalDataset(self.identifier, data={}, configuration=configuration, analysis_queries={}, report=None)
                    elif data_type == "wes" or data_type == "wgs":
                        if "wes" in self.configuration_files:
                            configuration = ckg_utils.get_configuration(self.configuration_files["wes"])
                        elif "wgs" in self.configuration_files:
                            configuration = ckg_utils.get_configuration(self.configuration_files["wgs"])
                        dataset = DNAseqDataset(self.identifier, dataset_type=data_type, data={}, configuration=configuration, analysis_queries={}, report=None)
                    elif data_type == "interactomics":
                        if "interactomics" in self.configuration_files:
                            configuration = ckg_utils.get_configuration(self.configuration_files["interactomics"])
                        dataset = InteractomicsDataset(self.identifier, data={}, configuration=configuration, analysis_queries={}, report=None)
                    elif data_type == "phosphoproteomics":
                        if "phosphoproteomics" in self.configuration_files:
                            configuration = ckg_utils.get_configuration(self.configuration_files["phosphoproteomics"])
                        dataset = PhosphoproteomicsDataset(self.identifier, data={}, configuration=configuration, analysis_queries={}, report=None)
                    elif data_type == "longitudinal_proteomics":
                        if "longitudinal_proteomics" in self.configuration_files:
                            configuration = ckg_utils.get_configuration(self.configuration_files["longitudinal_proteomics"])
                        dataset = LongitudinalProteomicsDataset(self.identifier, data={}, configuration=configuration, analysis_queries={}, report=None)

                    if dataset is not None:
                        dataset.generate_dataset()
                        self.update_dataset({data_type: dataset})

                if len(self.datasets) > 1:
                    configuration = None
                    if "multiomics" in self.configuration_files:
                        configuration = ckg_utils.get_configuration(self.configuration_files["multiomics"])
                    dataset = MultiOmicsDataset(self.identifier, data=self.datasets, configuration=configuration, report=None)
                    self.update_dataset({'multiomics': dataset})
                    self.append_data_type('multiomics')
            else:
                logger.error("Project {} could not be built. Error retrieving information for this project or no information associated to this project".format(self.identifier))
                print("Project {} could not be built. Error retrieving information for this project or no information associated to this project".format(self.identifier))

    def get_projects_overlap(self, project_info):
        if 'overlap' in project_info:
            self.overlap = project_info['overlap']
            if 'from' in self.overlap and 'to' in self.overlap:
                self.overlap = self.overlap[(self.overlap['from'] == self.identifier) | (self.overlap['to'] == self.identifier)]

    def get_similar_projects(self, project_info):
        if 'similarity' in project_info:
            self.similar_projects = project_info['similarity']
            if 'similarity_pearson' in self.similar_projects:
                self.similar_projects = self.similar_projects[self.similar_projects['similarity_pearson'] > 0.5]

    def generate_project_attributes_plot(self):
        project_df = self.to_dataframe()
        project_df = project_df.drop(['similar_projects', 'overlap'], axis=1)
        identifier = "Project info"
        title = "Project: {} information".format(self.name)
        plot = [viz.get_table(project_df, identifier, title)]

        return plot

    def generate_project_similarity_plots(self):
        plots = []
        identifier = "Similarities"
        title = "Similarities to other Projects"
        plots.append(viz.get_table(self.similar_projects, identifier+' table', title+' table'))
        plots.append(viz.get_sankey_plot(self.similar_projects, identifier, args={'source': 'current', 'target': 'other', 'weight': 'similarity_pearson', 'orientation': 'h', 'valueformat': '.0f', 'width': 800, 'height': 800, 'font': 12, 'title': title}))

        plots.append(self.get_similarity_network())

        return plots

    def generate_overlap_plots(self):
        plots = []
        identifier = "Overlap"
        title = "Protein Identification Overlap"
        plots.append(viz.get_table(self.overlap, identifier+' table', title+' table'))
        if self.overlap is not None:
            for i, row in self.overlap.iterrows():
                ntitle = title + ":\n" + row['project1_name'] +" - "+ row['project2_name'] +"(overlap similarity: " + str(row['similarity']) +")"
                plot = viz.plot_2_venn_diagram(row['from'], row['to'], row['project1_unique'], row['project2_unique'], row['intersection'], identifier=identifier+str(i), args={'title':ntitle})
                plots.append(plot)

        return plots

    def get_similarity_network_style(self):
        stylesheet = [{'selector': 'node',
                       'style': {'label': 'data(name)',
                                 'text-valign': 'center',
                                 'text-halign': 'center',
                                 'opacity': 0.8,
                                 'font-size': '12'}},
                      {'selector': 'edge',
                       'style': {'label': 'data(label)',
                                 'curve-style': 'bezier',
                                 'opacity': 0.7,
                                 'width': 0.4,
                                 'font-size': '5'}}]
        layout = {'name': 'cose',
                  'idealEdgeLength': 100,
                  'nodeOverlap': 20,
                  'refresh': 20,
                  #'fit': True,
                  #'padding': 30,
                  'randomize': False,
                  'componentSpacing': 100,
                  'nodeRepulsion': 400000,
                  'edgeElasticity': 100,
                  'nestingFactor': 5,
                  'gravity': 80,
                  'numIter': 1000,
                  'initialTemp': 200,
                  'coolingFactor': 0.95,
                  'minTemp': 1.0}

        return stylesheet, layout

    def get_similarity_network(self):
        plot = None
        try:
            query_path = os.path.join(cwd, self.queries_file)
            project_cypher = query_utils.read_queries(query_path)
            query = query_utils.get_query(project_cypher, query_id="projects_subgraph")
            list_projects = []
            driver = connector.getGraphDatabaseConnectionConfiguration()
            if self.similar_projects is not None:
                if "other_id" in self.similar_projects:
                    list_projects = self.similar_projects["other_id"].values.tolist()
                list_projects.append(self.identifier)
                list_projects = ",".join(['"{}"'.format(i) for i in list_projects])
                query = query.replace("LIST_PROJECTS", list_projects)
                path = connector.sendQuery(driver, query, parameters={})
                G = acore_utils.neo4j_path_to_networkx(path, key='path')
                args = {}
                style, layout = self.get_similarity_network_style()
                args['stylesheet'] = style
                args['layout'] = layout
                args['title'] = "Projects subgraph"
                net, mouseover = acore_utils.networkx_to_cytoscape(G)
                plot = viz.get_cytoscape_network(net, "projects_subgraph", args)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Error: {}. Reading queries from file {}: {}, file: {},line: {}".format(err, query_path, sys.exc_info(), fname, exc_tb.tb_lineno))

        return plot

    def generate_knowledge(self):
        nodes = {}
        relationships = {}
        keep_nodes = []
        kn = knowledge.ProjectKnowledge(identifier=self.identifier, data=self.to_dict(), nodes={self.name: {'id': '#0', 'type': 'Project'}}, relationships={}, colors={}, graph=None, report={})
        kn.generate_knowledge()
        nodes.update(kn.nodes)
        keep_nodes = kn.keep_nodes
        relationships.update(kn.relationships)
        types = ["clinical", "proteomics", "interactomics", "phosphoproteomics", "longitudinal_proteomics", "wes", "wgs", "rnaseq", "multiomics"]
        for dataset_type in types:
            if dataset_type in self.datasets:
                dataset = self.datasets[dataset_type]
                if dataset_type != "multiomics":
                    kn = dataset.generate_knowledge()
                else:
                    kn = dataset.generate_knowledge(nodes=nodes)
                    #kn.reduce_to_subgraph(nodes.keys())
                nodes.update(kn.nodes)

                relationships.update(kn.relationships)

        self.knowledge = knowledge.Knowledge(self.identifier, {'name': self.name}, nodes=nodes, relationships=relationships, keep_nodes=keep_nodes)

    def generate_project_info_report(self):
        report = rp.Report(identifier="project_info")

        plots = self.generate_project_attributes_plot()
        plots.extend(self.generate_project_similarity_plots())
        plots.extend(self.generate_overlap_plots())

        report.plots = {("Project info", "Project Information"): plots}

        return report

    def generate_report(self):
        if len(self.report) == 0:
            project_report = self.generate_project_info_report()
            self.update_report({"Project information": project_report})
            for dataset_type in self.data_types:
                dataset = self.get_dataset(dataset_type)
                if dataset is not None:
                    dataset.generate_report()
            self.generate_knowledge()
            self.knowledge.generate_report()
            self.knowledge.generate_report(visualizations=["network", "sankey"])
            self.save_project_report()
            self.save_project()
            self.save_project_datasets_data()
            self.notify_project_ready()
            self.download_project()

    def notify_project_ready(self, message_type='slack'):
        message = "Report for project "+str(self.name)+" is ready: check it out at http://localhost:8050/apps/project/"+str(self.identifier)
        subject = 'Report ready '+self.identifier
        message_from = "alsantosdel"
        message_to = "albsantosdel" #self.responsible_email
        if message_type == 'slack':
            utils.send_message_to_slack_webhook(message, message_to)
        else:
            utils.send_email(message, subject, message_from, message_to)

    def empty_report(self):
        self.report = {}

    def save_project_report(self):
        start = time.time()
        directory = self.get_report_directory()
        for report_name in self.report:
            report = self.report[report_name]
            dataset_dir = os.path.join(directory, report_name)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            report.save_report(dataset_dir)

        self.save_project_datasets_reports()
        self.knowledge.save_report(directory)
        print('save report', time.time() - start)

    def save_project_datasets_reports(self):
        start = time.time()
        directory = self.get_report_directory()
        for dataset_type in self.datasets:
            dataset = self.datasets[dataset_type]
            dataset_directory = os.path.join(directory, dataset_type)
            if isinstance(dataset, Dataset):
                dataset.save_report(dataset_directory)
                dataset = None
        print('save dataset report', time.time() - start)

    def save_project(self):
        directory = os.path.join(self.get_report_directory(), "Project information")
        if not os.path.isdir(directory):
            os.makedirs(directory)
        dt = h5.special_dtype(vlen=str)
        with h5.File(os.path.join(directory, "project_information_dataset.h5"), "w") as f:
            df_set = f.create_dataset("Project_information", (1,), dtype=dt, compression="gzip", chunks=True, data=self.to_json())

    def save_project_datasets_data(self):
        start = time.time()
        directory = self.get_report_directory()
        for dataset_type in self.datasets:
            dataset = self.datasets[dataset_type]
            dataset_directory = os.path.join(directory, dataset_type)
            if isinstance(dataset, Dataset):
                dataset.save_dataset(dataset_directory)
                dataset = None
        print('save datasets', time.time() - start)

    def show_report(self, environment):
        types = ["Project information", "clinical", "proteomics", "interactomics", "phosphoproteomics", "longitudinal_proteomics", "wes", "wgs", "rnaseq", "multiomics", "Knowledge Graph"]
        app_plots = defaultdict(list)
        for dataset in types:
            if dataset in self.report:
                report = self.report[dataset]
                if report is not None:
                    app_plots[dataset.upper()] = report.visualize_report(environment)
            elif dataset in self.datasets:
                report = self.datasets[dataset].report
                if report is not None:
                    app_plots[dataset.upper()] = report.visualize_report(environment)
            elif dataset == "Knowledge Graph":
                if self.knowledge is not None:
                    report = self.knowledge.report
                    if report is not None:
                        app_plots[dataset.upper()] = report.visualize_report(environment)

        return app_plots

    def download_project(self):
        directory = self.get_downloads_directory()
        self.download_project_report()
        self.download_project_datasets()
        utils.compress_directory(directory, directory, compression_format='zip')

    def download_project_report(self):
        directory = self.get_downloads_directory()
        project_sdrf = self.get_sdrf()
        if not project_sdrf.empty:
            project_sdrf.to_csv(os.path.join(directory, "{}.sdrf".format(self.identifier)), sep='\t', header=True, index=False, doublequote=None)

        for dataset in self.report:
            report = self.report[dataset]
            dataset_dir = os.path.join(directory, dataset)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            report.download_report(dataset_dir)
        for dataset in self.datasets:
            if isinstance(self.datasets[dataset], Dataset):
                report = self.datasets[dataset].report
                dataset_dir = os.path.join(directory, dataset)
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                report.download_report(dataset_dir)

        self.download_knowledge(os.path.join(directory, "Knowledge"))

    def download_knowledge(self, directory):
        report = self.knowledge.report
        if not os.path.exists(directory):
            os.makedirs(directory)

        report.download_report(directory)

    def download_project_datasets(self):
        directory = self.get_downloads_directory()
        for dataset_type in self.datasets:
            dataset = self.datasets[dataset_type]
            dataset_directory = os.path.join(directory, dataset_type)
            if isinstance(dataset, Dataset):
                dataset.save_dataset_to_file(dataset_directory)
