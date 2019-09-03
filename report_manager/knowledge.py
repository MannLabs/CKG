import os
import sys
import pandas as pd
import networkx as nx
import ckg_utils
import config.ckg_config as ckg_config
from graphdb_connector import connector
import logging
import logging.config

log_config = ckg_config.report_manager_log
logger = ckg_utils.setup_logging(log_config, key="knowledge")

class Knowledge:
    def __init__(self, identifier, data, nodes={}, relationships={}, queries_file=None, colors={}, graph=None):
        self._identifier = identifier
        self._data = data
        self._colors = {}
        self._nodes = nodes
        self._relationships = relationships
        self._queries_file = queries_file
        self._graph = graph
        self._default_color = '#878787'
        if len(colors) == 0:
            self._colors= {'Protein': '#3288bd', 
                           'Clinical_variable':'#1a9850',
                           'Drug':'#fdae61', 
                           'Disease':'#9e0142', 
                           'Pathway': '#abdda4', 
                           'Biological_process':'#e6f598', 
                           'Symptom':'#f46d43'}

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        self._identifier = identifier
        
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        
    @property
    def entities(self):
        return self._entities

    @entities.setter
    def entities(self, entities):
        self._entities = entities
        
    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes
        
    @property
    def relationships(self):
        return self._relationships

    @relationships.setter
    def relationships(self, relationships):
        self._relationships = relationships
        
    @property
    def queries_file(self):
        return self._queries_file

    @queries_file.setter
    def queries_file(self, queries_file):
        self._queries_file = queries_file
        
    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, colors):
        self._colors = colors
        
    @property
    def default_color(self):
        return self._default_color

    @default_color.setter
    def default_color(self, default_color):
        self._default_color = default_color
        
    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph
        
    def generate_knowledge_from_regulation(self, entity):
        nodes = {'Regulated':{'type':'default',  'colot':self.default_color}}
        relationships = {}
        color = self.colors[entity] if entity in self.colors else self.default_color
        if "regulated" in self.data:
            for n in self.data['regulated']:
                nodes.update({n : {'type': entity, 'color':color}})
                relationships.update({('Regulated', n): {'type':'is_regulated'}})
                
        return nodes, relationships
    
    def genreate_knowledge_from_correlation(self, entity_node1, entity_node2, filter):
        nodes = {}
        relationships = {}
        node1_color = self.colors[entity_node1] if entity_node1 in self.colors else self.default_color
        node2_color = self.colors[entity_node2] if entity_node2 in self.colors else self.default_color
        if 'correlation_correlation' in self.data:
            for i, row in self.data['correlation_correlation'].iterrows():
                if len(filter) > 0:
                    if row['node1'] not in filter or row['node2'] not in filter:
                        continue
                    nodes.update({row['node1']: {'type':entity_node1, 'color':node1_color}, row['node2'] : {'type':entity_node2, 'color':node2_color}})
                    relationships.update({(row['node1'], row['node2']):{'type': 'correlates', 'weight':row['weight']}})
        
        return nodes, relationships
        
    def generate_knowledge_from_wgcna(self, data, entity1, entity2):
        nodes = {}
        relationships = {}
        node1_color = self.colors[entity1] if entity1 in self.colors else self.default_color
        node2_color = self.colors[entity2] if entity2 in self.colors else self.default_color
        if 'features_per_module' in data:
            modules = data['features_per_module']
            for i,row in modules.iterrows():
                nodes.update({"ME"+row['modColor']: {'type':'Module', 'color':row['modColor']}, row['name'] : {'type':entity2, 'color':node2_color}})
                relationships.update({("ME"+row['modColor'], row['name']):{'type': 'belongs_to'}})
        if 'module_trait_cor' in data:
            correlations = data['module_trait_cor']
            correlations = correlations.set_index('index').stack().reset_index()
            for i,row in correlations.iterrows():
                nodes.update({row['level_1'] : {'type':entity1, 'color':node1_color}})
                relationships.update({(row['index'], row['level_1']):{'type': 'correlates', 'weight':row[0]}})
        
        return nodes, relationships
    
    def generate_knowledge_from_annotations(self, entity1, entity2, filter=None):
        nodes = {}
        relationships = {}
        node1_color = self.colors[entity1] if entity1 in self.colors else self.default_color
        node2_color = self.colors[entity2] if entity2 in self.colors else self.default_color
        if entity2.lower()+'_annotation' in self.data:
            for i, row in self.data[entity2.lower()+'_annotation'].iterrows():
                if len(filter) > 0:
                    if row['identifier'] not in filter or row['annotation'] not in filter:
                        continue
                    nodes.update({row['identifier']: {'type':entity1, 'color':node1_color}, row['annotation'] : {'type':entity2, 'color':node2_color}})
                    relationships.update({(row['identifier'], row['annotation']):{'type': 'is_annotated'}})
        
        return nodes, relationships
        
    
    def generate_knowledge_from_queries(self, entity, queries_results):
        nodes = {}
        relationships = {}
        for node2 in queries_results:
            node1_color = self.colors[entity] if entity in self.colors else self.default_color
            node2_color = self.colors[node2] if node2 in self.colors else self.default_color
            result = queries_results[node2]
            for i, row in result.iterrows():
                rel_type = row['type'] if 'type' in row else 'associated'
                nodes.update({row['node1']: {'type':entity, 'color':node1_color}, row['node2'] : {'type':node2, 'color':node2_color}})
                relationships.update({(row['node1'], row['node2']): {'type': rel_type, 'weight':row['weight']}})
        
        return nodes, relationships
    
    def send_query(self, query):
        driver = connector.getGraphDatabaseConnectionConfiguration()
        data = connector.getCursorData(driver, query)

        return data
    
    def query_data(self, replace):
        query_data = {}
        try:
            cwd = os.path.abspath(os.path.dirname(__file__))
            cypher_queries = ckg_utils.get_queries(os.path.join(cwd, self.queries_file))
            for query_name in cypher_queries:
                title = query_name.lower().replace('_',' ')
                query = cypher_queries[query_name]['query']
                for r,by in replace:
                    query = query.replace(r,by)
                query_data[title] = self.send_query(query)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Reading queries from file {}: {}, file: {},line: {}".format(self.queries_file, sys.exc_info(), fname, exc_tb.tb_lineno))

        return query_data
    
    def generate_cypher_nodes_list(self):
        nodes = ['"{}"'.format(n) for n in self.nodes.keys()]
        nodes = ",".join(nodes)
        return nodes
    
    def generate_knowledge_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes.items())
        G.add_edges_from(self.relationships.keys())
        nx.set_edge_attributes(G, self.relationships)
        self.graph = G
        
    def reduce_to_subgraph(self, nodes):
        valid_nodes = set(nodes).intersection(self.nodes)
        self.generate_knowledge_graph()
        graph = self.graph
        subgraph = graph.subgraph(valid_nodes)
        self.nodes = subgraph.nodes(data=True)
        self.relationships = subgraph.edges(data=True)
        self.graph = subgraph

class ProjectKnowledge(Knowledge):
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None):
        queries_file = 'queries/project_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph)
        
    def generate_knowledge(self):
        similarity_knowledge = self.generate_knowledge_from_similarity(entity='Project')
        self.nodes.update(similarity_knowledge[0])
        self.relationships.update(similarity_knowledge[1])
        
        queries_results = self.query_data(replace='PROJECTID', replace_with=self.nodes.keys())
        queries_knowledge = self.generate_knowledge_from_queries(entity='Project', queries_results=queries_results)
        self.nodes.update(queries_knowledge[0])
        self.relationships.update(queries_knowledge[1])
    
class ProteomicsKnowledge(Knowledge):
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None):
        queries_file = 'queries/proteomics_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph)
    
    def generate_knowledge(self):
        regulation_knowledge = self.generate_knowledge_from_regulation(entity='Protein')
        correlation_knowledge = self.genreate_knowledge_from_correlation('Protein', 'Protein', filter=regulation_knowledge[0].keys())
        self.nodes = regulation_knowledge[0]
        self.nodes.update(correlation_knowledge[0])
        self.relationships = regulation_knowledge[1]
        self.relationships.update(correlation_knowledge[1])
        
        nodes = self.generate_cypher_nodes_list()
        queries_results = self.query_data(replace=[('PROTEINIDS',nodes)])
        queries_knowledge = self.generate_knowledge_from_queries(entity='Protein', queries_results=queries_results)
        self.nodes.update(queries_knowledge[0])
        self.relationships.update(queries_knowledge[1])
        
class ClinicalKnowledge(Knowledge):
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None):
        queries_file = 'queries/clinical_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph)
        
    def generate_knowledge(self):
        regulation_knowledge = self.generate_knowledge_from_regulation(entity='Protein')
        correlation_knowledge = self.genreate_knowledge_from_correlation('Protein', 'Protein', filter=regulation_knowledge[0].keys())
        self.nodes = regulation_knowledge[0]
        self.nodes.update(correlation_knowledge[0])
        self.relationships = regulation_knowledge[1]
        self.relationships.update(correlation_knowledge[1])
        
        nodes = self.generate_cypher_nodes_list()
        queries_results = self.query_data(replace=[('PROJECTID', nodes)])
        queries_knowledge = self.generate_knowledge_from_queries(entity='Clinical', queries_results=queries_results)
        self.nodes.update(queries_knowledge[0])
        self.relationships.update(queries_knowledge[1])
        
class MultiOmicsKnowledge(Knowledge):
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None):
        queries_file = 'queries/multiomics_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph)
        
    def generate_knowledge(self):
        if 'wgcna_wgcna' in self.data:
            for dtype in self.data['wgcna_wgcna']:
                if dtype == 'wgcna-proteomics':
                    entity1 = 'Clinical_variable'
                    entity2 = 'Protein'
                    wgcna_knowledge = self.generate_knowledge_from_wgcna(self.data['wgcna_wgcna'][dtype], entity1, entity2)
                    self.nodes.update(wgcna_knowledge[0])
                    self.relationships.update(wgcna_knowledge[1])