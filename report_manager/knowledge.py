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
    
    def genreate_knowledge_from_correlation(self, entity_source, entity_target, filter):
        nodes = {}
        relationships = {}
        source_color = self.colors[entity_source] if entity_source in self.colors else self.default_color
        target_color = self.colors[entity_target] if entity_target in self.colors else self.default_color
        if 'correlation' in self.data:
            for row in self.data.iterrows():
                if len(filter) > 0:
                    if row['source'] not in filter or row['target'] not in filter:
                        continue
                    nodes.update({row['source']: {'type':entity_source, 'color':source_color}, row['target'] : {'type':entity_target, 'color':target_color}})
                    relationships.update({(row['source'], row['target']):{'type': 'correlates', 'weight':row['weight']}})
        
        return nodes, relationships
        
    def generate_knowledge_from_wgcna(self, entity1, entity2):
        nodes = {}
        relationships = {}
        source_color = self.colors[entity1] if entity1 in self.colors else self.default_color
        target_color = self.colors[entity2] if entity2 in self.colors else self.default_color
        #if 'correlation' in self.data:
        #    for row in data.iterrows():
        #        if len(filter) > 0:
        #            if row[args['source']] not in filter or row[args['target']] not in filter:
        #                continue
        #            nodes.update({row['source']: {'type':entity_source, 'color':source_color}, row['target'] : {'type':entity_target, 'color':target_color}})
        #            relationships.update({(row['source'], row['target']):{'type': 'correlates', 'weight':row['weight']})
        
        return nodes, relationships
    
    def generate_knowledge_from_queries(self, entity, queries_results):
        nodes = {}
        relationships = {}
        for target in queries_results:
            source_color = self.colors[entity] if entity in self.colors else self.default_color
            target_color = self.colors[target] if target in self.colors else self.default_color
            result = queries_results[target]
            nodes.update({result['source']: {'type':entity, 'color':source_color}, result['target'] : {'type':target, 'color':target_color}})
            relationships.update({(result['source'], result['target']): {'type': 'associated', 'weight':result['weight']}})
        
        return nodes, relationships
    
    def send_query(self, query):
        driver = connector.getGraphDatabaseConnectionConfiguration()
        data = connector.getCursorData(driver, query)

        return data
    
    def query_data(self, replace, replace_with):
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
        
    def generate_knowledge_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes.items())
        G.add_edges_from(self.relationships.keys())
        nx.set_edge_attributes(G, self.relationships)
        self.graph = G

class ProjectKnowledge(Knowledge):
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None):
        queries_file = 'queries/project_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph)
        
    def generate_knowledge(self):
        similarity_knowledge = self.generate_knowledge_from_similarity(entity='Protein')
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
        
        queries_results = self.query_data(replace='PROTEINIDS', replace_with=self.nodes.keys())
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
        
        queries_results = self.query_data(replace='PROJECTID', replace_with=self.nodes.keys())
        queries_knowledge = self.generate_knowledge_from_queries(entity='Clinical', queries_results=queries_results)
        self.nodes.update(queries_knowledge[0])
        self.relationships.update(queries_knowledge[1])
        
class MultiomicsKnowledge(Knowledge):
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None, entities_filter=[]):
        queries_file = 'queries/multiomics_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph)
        _entities_filter = entities_filter
        
    @property
    def entities_filter(self):
        return self._entities_filter

    @entities_filter.setter
    def entities_filter(self, entities_filter):
        self._entities_filter = entities_filter
        
    def generate_knowledge(self):
        if 'clinical' in self.data:
            for dtype in self.data:
                if dtype in ['proteomics', 'RNAseq']:
                    wgcna_knowledge = self.generate_knowledge_from_wgcna(entity1='clinical', entity2=dtype)
                    self.nodes.update(wgcna_knowledge[0])
                    self.relationships.update(wgcna_knowledge[1])