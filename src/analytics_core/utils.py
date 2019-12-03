import random
from Bio import Entrez
Entrez.email = 'alberto.santos@cpr.ku.dk'
from Bio import Medline
import re
import pandas as pd
import bs4 as bs
import requests, json
import networkx as nx
from networkx.readwrite import json_graph
from urllib import error

def generate_html(network):
        """
        This method gets the data structures supporting the nodes, edges,
        and options and updates the pyvis html template holding the visualization.

        :type name_html: str
        """
        # here, check if an href is present in the hover data
        use_link_template = False
        for n in network.nodes:
            title = n.get("title", None)
            if title:
                if "href" in title:
                    """
                    this tells the template to override default hover
                    mechanic, as the tooltip would move with the mouse
                    cursor which made interacting with hover data useless.
                    """
                    use_link_template = True
                    break
        template = network.template

        nodes, edges, height, width, options = network.get_network_data()
        network.html = template.render(height=height,
                                    width=width,
                                    nodes=nodes,
                                    edges=edges,
                                    options=options,
                                    use_DOT=network.use_DOT,
                                    dot_lang=network.dot_lang,
                                    widget=network.widget,
                                    bgcolor=network.bgcolor,
                                    conf=network.conf,
                                    tooltip_link=use_link_template)

def neo4j_path_to_networkx(paths, key='path'):
    regex = r"\(?(.+)\)\<?\-\>?\[\:(.+)\s\{.*\}\]\<?\-\>?\((.+)\)?"
    nodes = set()
    rels = set()
    for r in paths:
        if key is not None:
            path = str(r[key])
        matches = re.search(regex, path)
        if matches:
            source = matches.group(1)
            source_match = re.search(regex, source)
            if source_match:
                source = source_match.group(1)
                relationship = source_match.group(2)
                target = source_match.group(3)
                nodes.update([source, target])
                rels.add((source, target, relationship))
                source = target
            relationship = matches.group(2)
            target = matches.group(3)
            nodes.update([source, target])
            rels.add((source, target, relationship))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for s,t,label in rels:
        G.add_edge(s,t,label=label)

    return G

def neo4j_schema_to_networkx(schema):
    regex = r"\(?(.+)\)\<?\-\>?\[\:(.+)\s\{.*\}\]\<?\-\>?\((.+)\)"
    nodes = set()
    rels = set()
    if 'relationships' in schema[0]:
        relationships = schema[0]['relationships']
        for relationship in relationships:
            matches = re.search(regex, str(relationship))
            if matches:
                source = matches.group(1)
                source_match = re.search(regex, source)
                if source_match:
                    source = source_match.group(1)
                    relationship = source_match.group(2)
                    target = source_match.group(3)
                    nodes.update([source, target])
                    rels.add((source, target, relationship))
                    source = target
                relationship = matches.group(2)
                target = matches.group(3)
                nodes.update([source, target])
                rels.add((source, target, relationship))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    colors = dict(zip(nodes, get_hex_colors(len(nodes))))
    nx.set_node_attributes(G, colors, 'color')
    for s,t,label in rels:
        G.add_edge(s,t,label=label)
        
    return G
            
def networkx_to_cytoscape(graph):
    cy_graph = json_graph.cytoscape_data(graph)
    cy_nodes = cy_graph['elements']['nodes']
    cy_edges = cy_graph['elements']['edges']
    cy_elements = cy_nodes
    cy_elements.extend(cy_edges)
    mouseover_node = dict(graph.nodes(data=True))

    return cy_elements, mouseover_node

def networkx_to_gml(graph, path):
    nx.write_gml(graph, path)

def json_network_to_gml(graph_json, path):
    graph = json_network_to_networkx(graph_json)
    nx.write_gml(graph, path)

def json_network_to_networkx(graph_json):
    graph = json_graph.node_link_graph(graph_json)

    return graph

def get_clustergrammer_link(net, filename=None):
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO
    clustergrammer_url = 'http://amp.pharm.mssm.edu/clustergrammer/matrix_upload/'
    if filename is None:
        file_string = net.write_matrix_to_tsv()
        file_obj = StringIO(file_string)
        
        if 'filename' not in net.dat or net.dat['filename'] is None:
            fake_filename = 'Network.txt'
        else:
            fake_filename = net.dat['filename']
        
        r = requests.post(clustergrammer_url, files={'file': (fake_filename, file_obj)})
    else:
        file_obj = open(filename, 'r')
        r = requests.post(clustergrammer_url, files={'file': file_obj})
        
    link = r.text
    
    
    return link

def generator_to_dict(genvar):
    dictvar = {}
    for i,gen in enumerate(genvar):
            dictvar.update({n:i for n in gen})

    return dictvar

def parse_html(html_snippet):
    html_parsed = bs.BeautifulSoup(html_snippet)
    return html_parsed    

def hex2rgb(color):
    hex = color.lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2 ,4))
    rgba = rgb + (0.6,)
    return rgba

def get_rgb_colors(n):
    colors = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        colors.append((r,g,b)) 
    return colors

def get_hex_colors(n):
    initial_seed = 123
    colors = []
    for i in range(n):
        random.seed(initial_seed + i)
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        colors.append(color)

    return colors

def getMedlineAbstracts(idList):
    fields = {"TI":"title", "AU":"authors", "JT":"journal", "DP":"date", "MH":"keywords", "AB":"abstract", "PMID":"PMID"}
    pubmedUrl = "https://www.ncbi.nlm.nih.gov/pubmed/"
    abstracts = pd.DataFrame()
    try:
        handle = Entrez.efetch(db="pubmed", id=idList, rettype="medline", retmode="json")
        records = Medline.parse(handle)
        results = []
        for record in records:
            aux = {}
            for field in fields:
                if field in record:
                    aux[fields[field]] = record[field]
            if "PMID" in aux:
                aux["url"] = pubmedUrl + aux["PMID"]
            else:
                aux["url"] = ""
            
            results.append(aux)

        abstracts = pd.DataFrame.from_dict(results)
    except error.HTTPError as e:
        print("Request to Bio.Entrez failed. Error: {}".format(e))

    return abstracts