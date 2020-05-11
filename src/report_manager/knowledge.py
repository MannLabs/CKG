import os
import sys
import pandas as pd
import numpy as np
import ast
import networkx as nx
import ckg_utils
import config.ckg_config as ckg_config
import dash_cytoscape as cyto
from graphdb_connector import connector
from report_manager import report as rp
from analytics_core import utils
from analytics_core.viz import viz, color_list
from networkx.readwrite import json_graph

log_config = ckg_config.report_manager_log
logger = ckg_utils.setup_logging(log_config, key="knowledge")
cyto.load_extra_layouts()


class Knowledge:
    def __init__(self, identifier, data, nodes={}, relationships={}, queries_file=None, colors={}, graph=None, report={}):
        self._identifier = identifier
        self._data = data
        self._colors = {}
        self._nodes = nodes
        self._relationships = relationships
        self._queries_file = queries_file
        self._graph = graph
        self._report = report
        self._default_color = '#636363'
        self._colors = colors
        if len(colors) == 0:
            self._colors= {'Protein': '#1a9850',
                           'Clinical_variable': '#542788',
                           'Drug': '#c51b7d',
                           'Disease': '#b2182b',
                           'Pathway': '#762a83',
                           'Publication': '#b35806',
                           'Biological_process': '#e6f598',
                           'Symptom': '#f46d43',
                           'Project': '#3288bd',
                           'Complex': '#31a354'}

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
    def report(self):
        return self._report

    @report.setter
    def report(self, report):
        self._report = report
        
    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph
        
    def generate_knowledge_from_regulation(self, entity):
        nodes = {}
        relationships = {}
        color = self.colors[entity] if entity in self.colors else self.default_color
        if "regulated" in self.data:
            for n in self.data['regulated']:
                nodes.update({n : {'type': entity, 'color':color, 'parent':'Regulated'}})
                #relationships.update({('Regulated', n): {'type':'is_regulated', 'weight':1, 'source_color':self.default_color, 'target_color':color}})
                
        return nodes, relationships
    
    def genreate_knowledge_from_correlation(self, entity_node1, entity_node2, filter, cutoff=0.5):
        nodes = {}
        relationships = {}
        node1_color = self.colors[entity_node1] if entity_node1 in self.colors else self.default_color
        node2_color = self.colors[entity_node2] if entity_node2 in self.colors else self.default_color
        if 'correlation_correlation' in self.data:
            for i, row in self.data['correlation_correlation'].iterrows():
                if len(filter) > 0:
                    if row['node1'] not in filter or row['node2'] not in filter:
                        continue
                    if np.abs(row['weight']) >= cutoff:
                        nodes.update({row['node1']: {'type': entity_node1, 'color': node1_color}, row['node2']: {'type': entity_node2, 'color': node2_color}})
                        relationships.update({(row['node1'], row['node2']): {'type': 'correlates', 'weight': row['weight'], 'width': np.abs(row['weight']), 'source_color': node1_color, 'target_color': node2_color}})
            
        return nodes, relationships
        
    def generate_knowledge_from_wgcna(self, data, entity1, entity2, cutoff=0.2):
        nodes = {}
        relationships = {}
        color_dict = color_list.make_color_dict()
        node1_color = self.colors[entity1] if entity1 in self.colors else self.default_color
        node2_color = self.colors[entity2] if entity2 in self.colors else self.default_color
        if 'features_per_module' in data:
            modules = data['features_per_module']
            for i, row in modules.iterrows():
                nodes.update({"ME"+row['modColor']: {'type': 'Module', 'color': color_dict[row['modColor']], 'parent': 'Regulated'}, row['name']: {'type': entity2, 'color': node2_color, 'parent': "ME"+row['modColor']}})
                relationships.update({('Regulated', "ME"+row['modColor']): {'type': '', 'weight': 5, 'source_color': self.default_color, 'target_color': color_dict[row['modColor']]}})
                relationships.update({("ME"+row['modColor'], row['name']): {'type': 'CONTAINS', 'weight': 5, 'source_color': color_dict[row['modColor']], 'target_color': node2_color}})
        if 'module_trait_cor' in data and data['module_trait_cor'] is not None:
            correlations = data['module_trait_cor']
            if not correlations.index.is_numeric():
                correlations = correlations.reset_index()
            correlations = correlations.set_index('index').stack().reset_index()
            for i, row in correlations.iterrows():
                if np.abs(row[0]) >= cutoff:
                    nodes.update({row['level_1']: {'type': entity1, 'color': node1_color}})
                    relationships.update({(row['index'], row['level_1']): {'type': 'correlates', 'weight': row[0], 'width': row[0], 'source_color': color_dict[row['index'].replace('ME', '')], 'target_color': node1_color}})
            
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
                    relationships.update({(row['identifier'], row['annotation']):{'type': 'is_annotated','source_color':node1_color, 'target_color':node2_color}})
        
        return nodes, relationships
    
    def generate_knowledge_from_similarity(self, entity='Project'):
        nodes = {}
        relationships = {}
        node_color = self.colors[entity] if entity in self.colors else self.default_color
        if 'similar_projects' in self.data:
            similar_projects = pd.DataFrame.from_dict(self.data['similar_projects'])
            for i, row in similar_projects.iterrows():
                nodes.update({row['other']: {'type': entity, 'color': node_color}})
                relationships.update({(row['current'], row['other']): {'type': 'is_similar', 'weight': row['similarity_pearson'], 'width': row['similarity_pearson'], 'source_color': node_color, 'target_color': node_color}})
        
        return nodes, relationships
    
    def generate_knowledge_from_queries(self, entity, queries_results):
        nodes = {}
        relationships = {}
        for node2 in queries_results:
            node1_color = self.colors[entity] if entity in self.colors else self.default_color
            node2_color = self.colors[node2] if node2 in self.colors else self.default_color
            nodes.update({node2: {'color': node2_color, 'type': 'Group'}})
            result = queries_results[node2]
            for i, row in result.iterrows():
                rel_type = row['type'] if 'type' in row else 'associated'
                weight = row['weight'] if 'weight' in row else 5
                nodes.update({row['node1']: {'type': entity, 'color': node1_color}, row['node2'].replace("'", "").title(): {'type': node2, 'color': node2_color, 'parent': node2}})
                relationships.update({(row['node1'], row['node2'].replace("'", "").title()): {'type': rel_type, 'weight': weight, 'width': weight, 'source_color': node1_color, 'target_color': node2_color}})
                relationships.update({(row['node2'].replace("'", "").title(), node2): {'type': 'is_a', 'weight': 5, 'width': 5, 'source_color': node2_color, 'target_color': node2_color}})
        
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
            if cypher_queries is not None:
                for query_name in cypher_queries:
                    if 'query_type' in cypher_queries[query_name]:
                        if cypher_queries[query_name]['query_type'] == 'knowledge_report':
                            query = cypher_queries[query_name]['query']
                            for r, by in replace:
                                query = query.replace(r, by)
                            query_data[query_name] = self.send_query(query)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Reading queries from file {}: {}, file: {},line: {}, err: {}".format(self.queries_file, sys.exc_info(), fname, exc_tb.tb_lineno, err))

        return query_data
    
    def generate_cypher_nodes_list(self):
        nodes = ['"{}"'.format(n) for n in self.nodes.keys()]
        nodes = ",".join(nodes)
        return nodes
    
    def generate_knowledge_graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.items())
        G.add_edges_from(self.relationships.keys())
        nx.set_edge_attributes(G, self.relationships)
        self.graph = G
        
    def reduce_to_subgraph(self, nodes):
        valid_nodes = set(nodes).intersection(list(self.nodes.keys()))
        valid_nodes.add("Regulated")
        aux = set()
        self.generate_knowledge_graph()
        for n in valid_nodes:
            if n in self.nodes:
                for n1, n2,attr in self.graph.out_edges(n, data=True):
                    aux.add(n1)
                    aux.add(n2)
                for n1,n2,attr in self.graph.in_edges(n, data=True):
                    aux.add(n1)
                    aux.add(n2)
        remove = set(self.nodes.keys()).difference(aux.union(valid_nodes))
        self.graph.remove_nodes_from(list(remove))
        self.nodes = dict(self.graph.nodes(data=True))
        self.relationships = {(a,b):c for a,b,c in self.graph.edges(data=True)}
        
    def get_knowledge_graph_plot(self):
        if self.graph is None:
            self.generate_knowledge_graph()
        args = {'title': 'Project {} Knowledge Graph'.format(self.data['name']),
                'node_properties': {},
                'width': 2600,
                'height': 2600, 
                'maxLinkWidth': 7,
                'maxRadius': 20}
        color_selector = "{'selector': '[name = \"KEY\"]', 'style': {'font-size': 10, 'background-color':'VALUE','width': 50,'height': 50,'background-image':'/assets/graph_icons/ENTITY.png','background-fit': 'cover','opacity':OPACITY}}"
        stylesheet=[{'selector': 'node', 'style': {'label': 'data(name)', 'z-index': 9999}}, 
                    {'selector':'edge','style':{'label':'data(type)',
                                                'curve-style': 'unbundled-bezier', 
                                                'control-point-distance': '20px', 
                                                'control-point-weight': '0.7',
                                                'z-index': 5000, 
                                                'line-color': '#bdbdbd', 
                                                'opacity':0.2,
                                                'font-size':'7px'}}]
        layout = {'name': 'circle'}
        
        #stylesheet.extend([{'selector':'[weight < 0]', 'style':{'line-color':'#3288bd'}},{'selector':'[width > 0]', 'style':{'line-color':'#d73027'}}])
        for n in self.nodes:
            color = self.nodes[n]['color']
            image = self.nodes[n]['type']
            opacity = 0.3 if image == 'Module' or image == 'Group' else 1
            stylesheet.append(ast.literal_eval(color_selector.replace("KEY", n.replace("'","")).replace("VALUE",color).replace("ENTITY",image).replace("OPACITY",str(opacity))))
        stylesheet.extend([{'selector':'[weight < 0]', 'style':{'line-color':'#4add1'}},{'selector':'[weight > 0]', 'style':{'line-color':'#d6604d'}}])
        args['stylesheet'] = stylesheet
        args['layout'] = layout
        
        nodes_table, edges_table = viz.network_to_tables(self.graph)
        nodes_fig_table = viz.get_table(nodes_table, identifier=self.identifier+"_nodes_table", title="Nodes table")
        edges_fig_table = viz.get_table(edges_table, identifier=self.identifier+"_edges_table", title="Edges table")
        cy_elements, mouseover_node = utils.networkx_to_cytoscape(self.graph)
        #args['mouseover_node'] = mouseover_node

        net = {"notebook":[cy_elements, stylesheet, layout], "app":viz.get_cytoscape_network(cy_elements, self.identifier, args), "net_tables":(nodes_fig_table, edges_fig_table), "net_json":json_graph.node_link_data(self.graph)}
        
        return net
    
    def generate_report(self, visualization='sankey'):
        report = rp.Report(identifier="knowledge")
        if visualization == 'network':
            plots = [self.get_knowledge_graph_plot()]
        elif visualization == 'sankey':
            if self.graph is None:
                self.generate_knowledge_graph()
            df = nx.to_pandas_edgelist(self.graph).fillna(1)
            plots = [viz.get_sankey_plot(df, self.identifier, args={'source':'source', 
                                                                             'target':'target', 
                                                                             'source_colors':'source_color', 
                                                                             'target_colors':'target_color',
                                                                             'hover':'type',
                                                                             'pad':10,
                                                                             'weight':'weight',
                                                                             'orientation': 'h', 
                                                                             'valueformat': '.0f', 
                                                                             'width':1600, 
                                                                             'height':2200, 
                                                                             'font':10, 
                                                                             'title':'Knowledge Graph'})]
        report.plots = {("Knowledge Graph","Knowledge Graph"): plots}
        self.report = report
    
    def save_report(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(os.path.join(directory, "Knowledge")):
            os.makedirs(os.path.join(directory, "Knowledge"))
        self.report.save_report(directory=os.path.join(directory, "Knowledge"))

class ProjectKnowledge(Knowledge):
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None, report={}):
        queries_file = 'queries/project_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph, report=report)
        
    def generate_knowledge(self):
        similarity_knowledge = self.generate_knowledge_from_similarity(entity='Project')
        self.nodes.update(similarity_knowledge[0])
        self.relationships.update(similarity_knowledge[1])
        
        self.relationships.update({(self.data['name'], 'Regulated'): {'type': 'has', 'weight':5, 'width':5, 'source_color':self.colors['Project'], 'target_color':self.default_color}})
        
        queries_results = self.query_data(replace=[('PROJECTID',self.identifier)])
        queries_knowledge = self.generate_knowledge_from_queries(entity='Project', queries_results=queries_results)
        self.nodes.update(queries_knowledge[0])
        self.relationships.update(queries_knowledge[1])
    
class ProteomicsKnowledge(Knowledge):
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None, report={}):
        queries_file = 'queries/proteomics_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph, report=report)
    
    def generate_knowledge(self):
        regulation_knowledge = self.generate_knowledge_from_regulation(entity='Protein')
        #correlation_knowledge = self.genreate_knowledge_from_correlation('Protein', 'Protein', filter=regulation_knowledge[0].keys())
        self.nodes = regulation_knowledge[0]
        #self.nodes.update(correlation_knowledge[0])
        self.relationships = regulation_knowledge[1]
        #self.relationships.update(correlation_knowledge[1])
        nodes = self.generate_cypher_nodes_list()
        limit_count = 3 if len(nodes)>10 else 1
        queries_results = self.query_data(replace=[('PROTEINIDS',nodes), ('PROJECTID', self.identifier), ('LIMIT_COUNT', str(limit_count))])
        queries_knowledge = self.generate_knowledge_from_queries(entity='Protein', queries_results=queries_results)
        self.nodes.update(queries_knowledge[0])
        self.relationships.update(queries_knowledge[1])
        
class ClinicalKnowledge(Knowledge):
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None, report={}):
        queries_file = 'queries/clinical_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph, report=report)
        
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
    
    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None, report={}):
        queries_file = 'queries/multiomics_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph, report=report)
        
    def generate_knowledge(self):
        if 'wgcna_wgcna' in self.data:
            for dtype in self.data['wgcna_wgcna']:
                if dtype == 'wgcna-proteomics':
                    entity1 = 'Clinical_variable'
                    entity2 = 'Protein'
                    wgcna_knowledge = self.generate_knowledge_from_wgcna(self.data['wgcna_wgcna'][dtype], entity1, entity2)
                    self.nodes.update(wgcna_knowledge[0])
                    self.relationships.update(wgcna_knowledge[1])