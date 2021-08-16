import os
import sys
import re
import pandas as pd
import numpy as np
import ast
from operator import itemgetter
import networkx as nx
from ckg import ckg_utils
import dash_cytoscape as cyto
from ckg.graphdb_connector import connector
from ckg.report_manager import report as rp
from ckg.analytics_core import utils
from ckg.analytics_core.viz import viz, color_list
from networkx.readwrite import json_graph

ckg_config = ckg_utils.read_ckg_config()
log_config = ckg_config['report_manager_log']
logger = ckg_utils.setup_logging(log_config, key="knowledge")
cyto.load_extra_layouts()


class Knowledge:
    def __init__(self, identifier, data, focus_on="Protein", nodes={}, relationships={}, queries_file=None, keep_nodes=[], colors={}, graph=None, report={}):
        self._identifier = identifier
        self._data = data
        self._focus_on = focus_on
        self._colors = {}
        self._nodes = nodes
        self._relationships = relationships
        self._queries_file = queries_file
        self._graph = graph
        self._report = report
        self._default_color = '#636363'
        self._entities = ["Protein", "Disease", "Drug", "Pathway", "Biological_process", "Complex", "Publication", "Tissue", "Metabolite", "Phenotype"]
        self.remove_entity(self._focus_on)
        self._colors = colors
        self._keep_nodes = keep_nodes
        if len(colors) == 0:
            self._colors = {'Protein': '#756bb1',
                            'Clinical_variable': '#542788',
                            'Drug': '#c51b7d',
                            'Tissue': '#66c2a5',
                            'Disease': '#b2182b',
                            'Pathway': '#0570b0',
                            'Publication': '#b35806',
                            'Biological_process': '#e6f598',
                            'Metabolite': '#f46d43',
                            'Phenotype': '#ff7f00',
                            'Project': '#3288bd',
                            'Complex': '#31a354',
                            'upregulated': '#d53e4f',
                            'downregulated': '#3288bd'
                            }

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
    def focus_on(self):
        return self._focus_on

    @focus_on.setter
    def focus_on(self, focus_on):
        self._focus_on = focus_on

    @property
    def entities(self):
        return self._entities

    @entities.setter
    def entities(self, entities):
        self._entities = entities
        
    def remove_entity(self, entity):
        if entity in self._entities:
            self._entities.remove(entity)

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    def update_nodes(self, nodes):
        self._nodes.update(nodes)

    @property
    def relationships(self):
        return self._relationships

    @relationships.setter
    def relationships(self, relationships):
        self._relationships = relationships

    def update_relationships(self, relationships):
        self._relationships.update(relationships)

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

    @property
    def keep_nodes(self):
        return self._keep_nodes

    @keep_nodes.setter
    def keep_nodes(self, node_ids):
        self._keep_nodes = node_ids

    def empty_graph(self):
        self.nodes = {}
        self.relationships = {}
        self.graph = None
        
    def get_nodes(self, query_type):
        nodes = set()
        for node in self.nodes:
            if "type" in self.nodes[node]:
                if self.nodes[node]["type"] == query_type:
                    nodes.add(node)
        return list(nodes)

    def generate_knowledge_from_regulation(self, entity):
        nodes = {}
        relationships = {}
        color = self.colors[entity] if entity in self.colors else self.default_color
        if "regulated" in self.data:
            for n in self.data['regulated']:
                if n not in ['sample', 'group', 'subject']:
                    nodes.update({n: {'type': entity, 'color': color}})
                    relationships.update({('Regulated', n): {'type': 'is_regulated', 'weight': 1, 'source_color': self.default_color, 'target_color': color}})

        return nodes, relationships

    def genreate_knowledge_from_correlation(self, entity_node1, entity_node2, filter, cutoff=0.5, label='correlation_correlation'):
        nodes = {}
        relationships = {}
        node1_color = self.colors[entity_node1] if entity_node1 in self.colors else self.default_color
        node2_color = self.colors[entity_node2] if entity_node2 in self.colors else self.default_color
        if label in self.data:
            for i, row in self.data[label].iterrows():
                if len(filter) > 0:
                    if row['node1'] not in filter or row['node2'] not in filter:
                        continue
                    if np.abs(row['weight']) >= cutoff:
                        #nodes.update({row['node1']: {'type': entity_node1, 'color': node1_color}, row['node2']: {'type': entity_node2, 'color': node2_color}})
                        relationships.update({(row['node1'], row['node2']): {'type': 'correlates', 'weight': row['weight'], 'width': np.abs(row['weight']), 'source_color': node1_color, 'target_color': node2_color}})

        return nodes, relationships

    def generate_knowledge_from_associations(self, df, name):
        nodes = {}
        relationships = {}
        node1_color = self.colors['Protein'] if 'Protein' in self.colors else self.default_color
        if 'literature' not in name:
            entity = name.split('_')[1].capitalize()
            node2_color = self.colors[entity] if entity in self.colors else self.default_color
            if 'Proteins' in df and entity in df:
                if 'score' not in df:
                    df['score'] = 1.0

                aux = df[['Proteins', entity, 'score']]
                for i, row in aux.iterrows():
                    proteins = row['Proteins'].split(';')
                    for p in proteins:
                        nodes.update({p: {'type': 'Protein', 'color': node1_color}, row[entity]: {'type': entity, 'color': node2_color}})
                        relationships.update({(p, row[entity]): {'type': 'associated_with', 'weight': 0.0, 'width': np.abs(row['score']), 'source_color': node1_color, 'target_color': node2_color}})
        else:
            if 'PMID' in df and 'Proteins' in df and 'Diseases' in df:
                aux = df[['PMID', 'Proteins', 'Diseases']]
                aux['PMID'] = aux['PMID'].astype(int).astype(str)
                node2_color = self.colors["Publication"] if "Publication" in self.colors else self.default_color
                node3_color = self.colors["Disease"] if "Disease" in self.colors else self.default_color
                for i, row in aux.iterrows():
                    proteins = row['Proteins']
                    if proteins is not None:
                        if isinstance(proteins, str):
                            proteins = proteins.split(';')
                        for p in proteins:
                            nodes.update({p: {'type': 'Protein', 'color': node1_color}, "PMID:"+row['PMID']: {'type': "Publication", 'color': node2_color}})
                            relationships.update({(p, "PMID:"+row['PMID']): {'type': 'mentioned_in_publication', 'weight': 0.0, 'width': 1.0, 'source_color': node1_color, 'target_color': node2_color}})
                    diseases = row['Diseases']
                    if diseases is not None:
                        if isinstance(diseases, str):
                            diseases = diseases.split(';')
                        for d in diseases:
                            nodes.update({d: {'type': 'Disease', 'color': node3_color}})
                            relationships.update({(d, "PMID:"+row['PMID']): {'type': 'mentioned_in_publication', 'weight': 0.0, 'width': 1.0, 'source_color': node3_color, 'target_color': node2_color}})

        return nodes, relationships

    def generate_knowledge_from_interactions(self, df, name):
        nodes = {}
        relationships = {}
        entity = name.split('_')[0].capitalize()
        if 'node1' in df and 'node2' in df and 'score' in df:
            for node1, node2, score in df[['node1', 'node2', 'score']].to_records():
                nodes.update({node1: {'type': entity, 'color': self.colors[entity]}, node2: {'type': entity, 'color': self.colors[entity]}})
                relationships.update({(node1, node2): {'type': 'interacts_with', 'weight': 0.0, 'width': score, 'source_color': self.colors[entity], 'target_color': self.colors[entity]}})

        return nodes, relationships

    def generate_knowledge_from_enrichment(self, data, name):
        nodes = {}
        relationships = {}
        entity = name.split('_')[0].capitalize()
        node1_color = self.colors[entity] if entity in self.colors else self.default_color
        if isinstance(data, pd.DataFrame):
            aux = data.copy()
            data = {'regulation': aux}
        for g in data:
            df = data[g]
            if 'terms' in df and 'identifiers' in df and 'padj' in df:
                aux = df[df.rejected]
                aux = aux[['terms', 'identifiers', 'padj']]
                for i, row in aux.iterrows():
                    ids = row['identifiers'].split(',')
                    if ids is not None:
                        for i in ids:
                            if 'Pathways' in name:
                                entity2 = 'Pathway'
                            elif 'processes' in name:
                                entity2 = 'Biological_process'

                            node2_color = self.colors[entity2] if entity2 in self.colors else self.default_color
                            nodes.update({i: {'type': entity, 'color': node1_color}, row['terms']: {'type': entity2, 'color': node2_color}})
                            relationships.update({(i, row['terms']): {'type': 'annotated_in', 'weight': 0.0, 'width': -np.log10(row['padj'])+1, 'source_color': node1_color, 'target_color': node2_color}})

        return nodes, relationships

    def generate_knowledge_from_dataframes(self):
        graph_rels = {}
        graph_nodes = {}
        for name in self.data:
            df = self.data[name]
            if isinstance(df, pd.DataFrame):
                df = df.dropna()
                if 'associations' in name:
                    nodes, rels = self.generate_knowledge_from_associations(df, name)
                    graph_nodes.update(nodes)
                    graph_rels.update(rels)
                elif 'interaction' in name:
                    nodes, rels = self.generate_knowledge_from_interactions(df, name)
                    graph_nodes.update(nodes)
                    graph_rels.update(rels)
                elif 'enrichment' in name:
                    nodes, rels = self.generate_knowledge_from_enrichment(df, name)
                    graph_nodes.update(nodes)
                    graph_rels.update(rels)
            elif isinstance(df, dict):
                nodes, rels = self.generate_knowledge_from_enrichment(df, name)
                graph_nodes.update(nodes)
                graph_rels.update(rels)

        return graph_nodes, graph_rels

    def generate_knowledge_from_wgcna(self, data, entity1, entity2, cutoff=0.2):
        nodes = {}
        relationships = {}
        color_dict = color_list.make_color_dict()
        node1_color = self.colors[entity1] if entity1 in self.colors else self.default_color
        node2_color = self.colors[entity2] if entity2 in self.colors else self.default_color
        if 'features_per_module' in data:
            modules = data['features_per_module']
            for i, row in modules.iterrows():
                nodes.update({"ME"+row['modColor']: {'type': 'Module', 'color': color_dict[row['modColor']]}, row['name']: {'type': entity2, 'color': node2_color}})
                relationships.update({('Regulated', "ME"+row['modColor']): {'type': '', 'weight': 5, 'width': 1.0, 'source_color': self.default_color, 'target_color': color_dict[row['modColor']]}})
                relationships.update({("ME"+row['modColor'], row['name']): {'type': 'CONTAINS', 'weight': 5, 'width': 1.0, 'source_color': color_dict[row['modColor']], 'target_color': node2_color}})
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

    def generate_knowledge_from_edgelist(self, edgelist, entity1, entity2, source, target, rtype, weight, source_attr=[], target_attr=[]):
        nodes = {}
        relationships = {}
        node1_color = self.colors[entity1] if entity1 in self.colors else self.default_color
        node2_color = self.colors[entity2] if entity2 in self.colors else self.default_color
        edgelist[source] = edgelist[source].astype(str)
        edgelist[target] = edgelist[target].astype(str)
        for i, row in edgelist.iterrows():
            attr1 = {'type': entity1, 'color': node1_color}
            attr1.update({c: row[c] for c in source_attr if c in row})
            attr2 = {'type': entity2, 'color': node2_color}
            attr2.update({c: row[c] for c in target_attr if c in row})
            
            nodes.update({row[source].replace("'", ""): attr1, row[target].replace("'", ""): attr2})
            relationships.update({(row[source].replace("'", ""), row[target].replace("'", "")): {'type': rtype, 'source_color': node1_color, 'target_color': node2_color, 'weight': row[weight]}})

        self.update_nodes(nodes)
        self.update_relationships(relationships)
        

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
                    nodes.update({row['identifier']: {'type': entity1, 'color': node1_color}, row['annotation']: {'type': entity2, 'color': node2_color}})
                    relationships.update({(row['identifier'], row['annotation']): {'type': 'is_annotated', 'source_color': node1_color, 'target_color': node2_color}})

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
                nodes.update({row['node1']: {'type': entity, 'color': node1_color}, row['node2'].replace("'", ""): {'type': node2, 'color': node2_color}})
                relationships.update({(row['node1'], row['node2'].replace("'", "")): {'type': rel_type, 'weight': weight, 'width': weight, 'source_color': node1_color, 'target_color': node2_color}})
                relationships.update({(row['node2'].replace("'", ""), node2): {'type': 'is_a', 'weight': 5, 'width': 1.0, 'source_color': node2_color, 'target_color': node2_color}})

        return nodes, relationships

    def send_query(self, query):
        driver = connector.getGraphDatabaseConnectionConfiguration()
        data = connector.getCursorData(driver, query)

        return data

    def query_data(self, replace=[]):
        query_data = {}
        try:
            cwd = os.path.dirname(os.path.abspath(__file__))
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

    def annotate_list(self, query_list, entity_type, attribute='name', queries_file=None, diseases=[], entities=None):
        self.empty_graph()
        if queries_file is None:
            queries_file = 'queries/knowledge_annotation.yml'

        if entities is None:
            entities = self.entities

        if diseases is None or len(diseases) < 1:
            replace_by = ('DISEASE_COND', '')
        else:
            replace_by = ('DISEASE_COND', 'OR d.name IN {} AND r.score > 1.5'.format(diseases))
            self.keep_nodes.extend(diseases)

        query_data = []
        drugs = []
        targets = []
        q = 'NA'
        try:
            if len(query_list) > 1:
                cwd = os.path.dirname(os.path.abspath(__file__))
                cypher_queries = ckg_utils.get_queries(os.path.join(cwd, queries_file))
                if cypher_queries is not None:
                    if entity_type.capitalize() in cypher_queries:
                        queries = cypher_queries[entity_type.capitalize()]
                        for query_name in queries:
                            involved_nodes = queries[query_name]['involved_nodes']
                            if len(set(involved_nodes).intersection(entities)) > 0 or query_name.capitalize() == entity_type.capitalize():
                                query = queries[query_name]['query']
                                q = 'NA'
                                for q in query.split(';')[:-1]:
                                    if attribute is None:
                                        matches = re.finditer(r'(\w+).ATTRIBUTE', q)
                                        for matchNum, match in enumerate(matches, start=1):
                                            var = match.group(1)
                                            q = q.format(query_list=query_list).replace("ATTRIBUTE", 'name+"~"+{}.id'.format(var)).replace(replace_by[0], replace_by[1]).replace('DISEASES', str(diseases)).replace('DRUGS', str(drugs)).replace('TARGETS', str(targets))
                                        else:
                                            q = q.format(query_list=query_list).replace(replace_by[0], replace_by[1]).replace('DISEASES', str(diseases)).replace('DRUGS', str(drugs)).replace('TARGETS', str(targets))
                                    else:
                                        q = q.format(query_list=query_list).replace("ATTRIBUTE", attribute).replace(replace_by[0], replace_by[1]).replace('DISEASES', str(diseases)).replace('DRUGS', str(drugs)).replace('TARGETS', str(targets))
                                    data = self.send_query(q)
                                    if not data.empty:
                                        if query_name == 'disease' and len(diseases) < 1:
                                            diseases = data['target'].unique().tolist()
                                        elif query_name == 'drug':
                                            drugs = data['target'].unique().tolist()
                                        elif query_name == 'target':
                                            targets = data['target'].dropna().unique().tolist()
                                        query_data.append(data)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Error annotating list. Query: {} from file {}: {}, file: {},line: {}, err: {}".format(q, queries_file, sys.exc_info(), fname, exc_tb.tb_lineno, err))
            print("Error annotating list. Query: {} from file {}: {}, file: {},line: {}, err: {}".format(q, queries_file, sys.exc_info(), fname, exc_tb.tb_lineno, err))

        if len(query_data) > 0:
            self.data = pd.DataFrame().append(query_data)
            for df in query_data:
                if 'source_type' in df and 'target_type' in df:
                    entity1 = df['source_type'][0][0]
                    entity2 = df['target_type'][0][0]
                    if 'rel_type' in df:
                        assoc_type = df['rel_type'][0]
                    else:
                        assoc_type = 'relationship'
                    
                    if 'weight' in df:
                        df['weight'] = df['weight'].fillna(0.5)
                    else:
                        df['weight'] = 0.5
                self.generate_knowledge_from_edgelist(df, entity1, entity2, source='source', target='target', rtype=assoc_type, weight='weight')
                

    def generate_cypher_nodes_list(self):
        nodes = ['"{}"'.format(n) for n in self.nodes.keys()]
        nodes = ",".join(nodes)
        return nodes

    def generate_knowledge_graph(self, summarize=True, method='betweenness', inplace=True, num_nodes=15):
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.items())
        G.add_edges_from(self.relationships.keys())
        nx.set_edge_attributes(G, self.relationships)
        selected_nodes = []
        if summarize and len(G.nodes()) > 1:
            centrality = None
            if method == 'betweenness':
                k = None if len(G.nodes()) < 15000 else 15000
                centrality = nx.betweenness_centrality(G, k=k, weight='weight', normalized=False)
            elif method == 'closeness':
                centrality = nx.closeness_centrality(G, u=None, distance='weight', wf_improved=True)
            elif method == 'pagerank':
                centrality = nx.pagerank(G, alpha=0.95, weight='weight')
            elif method == 'degree':
                centrality = nx.degree_centrality(G)

            if centrality is not None:
                nx.set_node_attributes(G, centrality, 'centrality')
                sorted_centrality = sorted(centrality.items(), key=itemgetter(1), reverse=True)
                for node_type in self.entities:
                    nodes = [x for x, y in G.nodes(data=True) if 'type' in y and y['type'] == node_type and x not in self.keep_nodes]
                    selected_nodes.extend([n for n, c in sorted_centrality if n in nodes][num_nodes:])

                if len(selected_nodes) > 0:
                    G.remove_nodes_from(selected_nodes)
                    G.remove_nodes_from(list(nx.isolates(G)))
        if inplace:
            self.graph = G.copy()

        return G

    def reduce_to_subgraph(self, nodes, summarize=True):
        valid_nodes = set(nodes).intersection(list(self.nodes.keys()))
        valid_nodes.add("Regulated")
        aux = set()
        self.generate_knowledge_graph()
        for n in valid_nodes:
            if n in self.nodes:
                for n1, n2, attr in self.graph.out_edges(n, data=True):
                    aux.add(n1)
                    aux.add(n2)
                for n1, n2, attr in self.graph.in_edges(n, data=True):
                    aux.add(n1)
                    aux.add(n2)
        if self.graph is not None:
            remove = set(self.nodes.keys()).difference(aux.union(valid_nodes))
            self.graph.remove_nodes_from(list(remove))
            self.nodes = dict(self.graph.nodes(data=True))
            self.relationships = {(a, b): c for a, b, c in self.graph.edges(data=True)}

    def get_knowledge_graph_plot(self, graph=None):
        if graph is None:
            graph = self.graph.copy()
        title = 'Project {} Knowledge Graph'.format(self.identifier)
        if self.data is not None:
            if 'name' in self.data:
                title = 'Project {} Knowledge Graph'.format(self.data['name'])

        args = {'title': title,
                'node_properties': {},
                'width': 2000,
                'height': 2000,
                'maxLinkWidth': 7,
                'maxRadius': 20}
        color_selector = "{'selector': '[name = \"KEY\"]', 'style': {'font-size': '7px', 'text-opacity': 0.8, 'background-color':'VALUE','width': 50,'height': 50,'background-image':'/assets/graph_icons/ENTITY.png','background-fit': 'cover','opacity':OPACITY}}"
        stylesheet = [{'selector': 'node', 'style': {'label': 'data(name)', 'opacity': 0.7}},
                      {'selector': 'edge', 'style': {'label': 'data(type)',
                                                     'curve-style': 'unbundled-bezier',
                                                     'control-point-distance': '30px',
                                                     'control-point-weight': '0.7',
                                                     'z-index': 5000,
                                                     'line-color': '#bdbdbd',
                                                     'opacity': 0.2,
                                                     'font-size': '2.5px',
                                                     'text-opacity': 1,
                                                     'font-style': "normal",
                                                     'font-weight': "normal"}}]
        layout = {'name': 'cose',
                  'idealEdgeLength': 100,
                  'nodeOverlap': 20,
                  'refresh': 20,
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

        stylesheet.extend([{'selector': '[weight < 0]', 'style': {'line-color': '#3288bd'}}, {'selector': '[weight > 0]', 'style': {'line-color': '#d73027'}}])
        for n, attr in graph.nodes(data=True):
            color = self.default_color
            image = ''
            if 'color' in attr:
                color = attr['color']
            if 'type' in attr:
                image = attr['type']
            opacity = 0.3 if image == 'Module' or image == 'Group' else 1
            stylesheet.append(ast.literal_eval(color_selector.replace("KEY", n.replace("'", "")).replace("VALUE", color).replace("ENTITY", image).replace("OPACITY", str(opacity))))
        stylesheet.extend([{'selector': 'node', 'style': {'width': 'mapData(centrality, 0, 1, 15, 30)', 'height': 'mapData(centrality, 0, 1, 15, 30)'}}])
        args['stylesheet'] = stylesheet
        args['layout'] = layout

        if graph.has_node('Regulated'):
            graph.remove_node('Regulated')
        nodes_table, edges_table = viz.network_to_tables(graph, source='node1', target='node2')
        nodes_fig_table = viz.get_table(nodes_table, identifier=self.identifier + "_nodes_table", args={'title': "Nodes table"})
        edges_fig_table = viz.get_table(edges_table, identifier=self.identifier + "_edges_table", args={'title': "Edges table"})
        cy_elements, mouseover_node = utils.networkx_to_cytoscape(graph)
        #args['mouseover_node'] = mouseover_node

        net = {"notebook": [cy_elements, stylesheet, layout], "app": viz.get_cytoscape_network(cy_elements, self.identifier, args), "net_tables": (nodes_table, edges_table), "net_tables_viz": (nodes_fig_table, edges_fig_table), "net_json": json_graph.node_link_data(graph)}

        return net

    def generate_knowledge_sankey_plot(self, graph=None):
        remove_edges = []
        if graph is None:
            graph = self.graph.copy()
        new_type_edges = {}
        new_type_nodes = {}
        for n1, n2 in graph.edges():
            if graph.nodes[n1]['type'] == graph.nodes[n2]['type']:
                remove_edges.append((n1, n2))
            else:
                if graph.nodes[n1]['type'] in self.entities:
                    color = graph.nodes[n1]['color']
                    new_type_edges.update({(n1, graph.nodes[n1]['type']): {'type': 'is_a', 'weight': 0.0, 'width': 1.0, 'source_color': color, 'target_color': self.colors[graph.nodes[n1]['type']]}})
                    new_type_nodes.update({graph.nodes[n1]['type']: {'type': 'entity', 'color': self.colors[graph.nodes[n1]['type']]}})
                if graph.nodes[n2]['type'] in self.entities:
                    color = graph.nodes[n2]['color']
                    new_type_edges.update({(n2, graph.nodes[n2]['type']): {'type': 'is_a', 'weight': 0.0, 'width': 1.0, 'source_color': color, 'target_color': self.colors[graph.nodes[n2]['type']]}})
                    new_type_nodes.update({graph.nodes[n2]['type']: {'type': 'entity', 'color': self.colors[graph.nodes[n2]['type']]}})

        graph.remove_edges_from(remove_edges)
        graph.add_edges_from(new_type_edges.keys())
        nx.set_edge_attributes(graph, new_type_edges)
        graph.add_nodes_from(new_type_nodes.items())
        df = nx.to_pandas_edgelist(graph).fillna(0.5)
        plot = viz.get_sankey_plot(df, self.identifier, args={'source': 'source',
                                                                    'target': 'target',
                                                                    'source_colors': 'source_color',
                                                                    'target_colors': 'target_color',
                                                                    'hover': 'type',
                                                                    'pad': 10,
                                                                    'weight': 'width',
                                                                    'orientation': 'h',
                                                                    'valueformat': '.0f',
                                                                    'width': 1600,
                                                                    'height': 2200,
                                                                    'font': 10,
                                                                    'title':'Knowledge Graph'})

        return plot

    def generate_report(self, visualizations=['sankey'], summarize=True, method='betweenness', inplace=True, num_nodes=15):
        report = rp.Report(identifier="knowledge")
        plots = []
        G = None
        if self.graph is None:
            G = self.generate_knowledge_graph(summarize=summarize, method=method, inplace=inplace, num_nodes=num_nodes)

        for visualization in visualizations:
            if visualization == 'network':
                plots.append(self.get_knowledge_graph_plot(G))
            elif visualization == 'sankey':
                plots.append(self.generate_knowledge_sankey_plot(G))

        report.plots = {("Knowledge Graph", "Knowledge Graph"): plots}
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
        self.nodes.update({self.data["name"]: {"type":'project', 'color': self.colors['Project']}, "Regulated": {'type': "connector", 'color': self.default_color}})
        self.relationships.update({(self.data['name'], 'Regulated'): {'type': 'has', 'weight':5, 'width':1.0, 'source_color': self.colors['Project'], 'target_color': self.default_color}})
        queries_results = self.query_data(replace=[('PROJECTID',self.identifier)])
        queries_knowledge = self.generate_knowledge_from_queries(entity='Project', queries_results=queries_results)
        self.nodes.update(queries_knowledge[0])
        self.keep_nodes = list(queries_knowledge[0].keys())
        self.relationships.update(queries_knowledge[1])

class ProteomicsKnowledge(Knowledge):

    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None, report={}):
        queries_file = 'queries/proteomics_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph, report=report)

    def generate_knowledge(self):
        regulation_knowledge = self.generate_knowledge_from_regulation(entity='Protein')
        self.nodes = regulation_knowledge[0]
        self.relationships = regulation_knowledge[1]
        #self.nodes.update(correlation_knowledge[0])
        #self.relationships.update(correlation_knowledge[1])
        #nodes = self.generate_cypher_nodes_list()
        #limit_count = 3 if len(nodes)>10 else 1
        #queries_results = self.query_data(replace=[('PROTEINIDS',nodes), ('PROJECTID', self.identifier), ('LIMIT_COUNT', str(limit_count))])
        #queries_knowledge = self.generate_knowledge_from_queries(entity='Protein', queries_results=queries_results)
        #self.nodes.update(queries_knowledge[0])
        #self.relationships.update(queries_knowledge[1])
        df_knowledge = self.generate_knowledge_from_dataframes()
        self.nodes.update(df_knowledge[0])
        self.relationships.update(df_knowledge[1])        

class ClinicalKnowledge(Knowledge):

    def __init__(self, identifier, data, nodes={}, relationships={}, colors={}, graph=None, report={}):
        queries_file = 'queries/clinical_knowledge_cypher.yml'
        Knowledge.__init__(self, identifier, data=data, nodes=nodes, relationships=relationships, queries_file=queries_file, colors=colors, graph=graph, report=report)

    def generate_knowledge(self):
        regulation_knowledge = self.generate_knowledge_from_regulation(entity='Clinical_variable')
        #correlation_knowledge = self.genreate_knowledge_from_correlation('Clinical_variable', 'Clinical_variable', filter=regulation_knowledge[0].keys())
        self.nodes = regulation_knowledge[0]
        #self.nodes.update(correlation_knowledge[0])
        self.relationships = regulation_knowledge[1]
        #self.relationships.update(correlation_knowledge[1])

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
        elif 'clinical_correlation_multi_correlation' in self.data:
            label = 'clinical_correlation_multi_correlation'
            correlation_knowledge = self.genreate_knowledge_from_correlation('Protein', 'Clinical_variable', filter=self.nodes, label=label)
            self.nodes.update(correlation_knowledge[0])
            self.relationships.update(correlation_knowledge[1])
