import random
from Bio import Entrez, Medline
from collections import defaultdict
import pandas as pd
import io
import base64
import bs4 as bs
import dash_html_components as html
import requests
import networkx as nx
from networkx.readwrite import json_graph
from urllib import error

Entrez.email = 'alberto.santos@cpr.ku.dk' # TODO: This should probably be changed to the email of the person installing ckg?


def check_columns(df, cols):
    for col in cols:
        if col not in df:
            return False
    return True


def mpl_to_html_image(plot, width=800):
    buf = io.BytesIO()
    plot.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    figure = html.Img(src="data:image/png;base64,{}".format(data), width="800")

    return figure


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
    network.html = template.render(height=height, width=width, nodes=nodes, edges=edges, options=options, use_DOT=network.use_DOT, dot_lang=network.dot_lang,
                                   widget=network.widget, bgcolor=network.bgcolor, conf=network.conf, tooltip_link=use_link_template)


def append_to_list(mylist, myappend):
    if isinstance(myappend, list):
        mylist.extend(myappend)
    else:
        mylist.append(myappend)


def neo4j_path_to_networkx(paths, key='path'):
    nodes = set()
    rels = set()
    for path in paths:
        if key in path:
            relationships = path[key]
            if len(relationships) == 3:
                node1, rel, node2 = relationships
                if 'name' in node1:
                    source = node1['name']
                if 'name' in node2:
                    target = node2['name']

                nodes.update([source, target])
                rels.add((source, target, rel))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for s, t, label in rels:
        G.add_edge(s, t, label=label)

    return G


def neo4j_schema_to_networkx(schema):
    nodes = set()
    rels = set()
    if 'relationships' in schema[0]:
        relationships = schema[0]['relationships']
        for node1, rel, node2 in relationships:
            if 'name' in node1:
                source = node1['name']
            if 'name' in node2:
                target = node2['name']

            nodes.update([source, target])
            rels.add((source, target, rel))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    colors = dict(zip(nodes, get_hex_colors(len(nodes))))
    nx.set_node_attributes(G, colors, 'color')
    for s, t, label in rels:
        G.add_edge(s, t, label=label)

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


def networkx_to_neo4j_document(graph):
    graph_json = []
    seen_rels = set()
    for n, attr in graph.nodes(data=True):
        rels = defaultdict(list)
        attr.update({'id': n})
        for r in graph[n]:
            edge = graph[n][r]
            edge.update({'id': r})
            if 'type' in edge:
                rel_type = edge['type']
                if 'type' in graph.nodes()[r]:
                    edge['type'] = graph.nodes()[r]['type']
                if not (n, r, edge['type']) in seen_rels:
                    rels[rel_type].append(edge)
                    seen_rels.update({(n, r, edge['type']), (r, n, edge['type'])})
                    attr.update(rels)
        graph_json.append(attr)

    return graph_json


def json_network_to_gml(graph_json, path):
    graph = json_network_to_networkx(graph_json)
    with open(path, 'wb') as out:
        nx.write_gml(graph, out)


def networkx_to_graphml(graph, path):
    nx.write_graphml(graph, path)


def json_network_to_graphml(graph_json, path):
    graph = json_network_to_networkx(graph_json)
    with open(path, 'wb') as out:
        nx.write_graphml(graph, out)


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
    for i, gen in enumerate(genvar):
        dictvar.update({n: i for n in gen})

    return dictvar


def parse_html(html_snippet):
    html_parsed = bs.BeautifulSoup(html_snippet, 'html.parser')

    return html_parsed


def convert_html_to_dash(el, style=None):
    ALLOWED_CST = {'div', 'span', 'a', 'hr', 'br', 'p', 'b', 'i', 'u', 's', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ol', 'ul', 'li',
                   'em', 'strong', 'cite', 'tt', 'pre', 'small', 'big', 'center', 'blockquote', 'address', 'font', 'img',
                   'table', 'tr', 'td', 'caption', 'th', 'textarea', 'option'}

    def __extract_style(el):
        if not el.attrs.get("style"):
            return None

        return {k.strip(): v.strip() for k, v in [x.split(": ") for x in el.attrs["style"].split(";") if x != '']}

    if type(el) is str:
        return convert_html_to_dash(parse_html(el))
    if type(el) == bs.element.NavigableString:
        return str(el)
    else:
        name = el.name
        style = __extract_style(el) if style is None else style
        contents = [convert_html_to_dash(x) for x in el.contents]
        if name.title().lower() not in ALLOWED_CST:
            return contents[0] if len(contents) == 1 else html.Div(contents)
        return getattr(html, name.title())(contents, style=style)


def hex2rgb(color):
    hex = color.lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
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
        colors.append((r, g, b))
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
    fields = {"TI": "title", "AU": "authors", "JT": "journal", "DP": "date", "MH": "keywords", "AB": "abstract", "PMID": "PMID"}
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
    except error.URLError as e:
        print("URLError: Request to Bio.Entrez failed. Error: {}".format(e))
    except error.HTTPError as e:
        print("HTTPError: Request to Bio.Entrez failed. Error: {}".format(e))
    except Exception as e:
        print("Request to Bio.Entrez failed. Error: {}".format(e))

    return abstracts
