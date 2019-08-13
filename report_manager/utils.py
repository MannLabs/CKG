import os
import dash_html_components as html
import dash_core_components as dcc
from datetime import date
import bs4 as bs
import random
from Bio import Entrez
Entrez.email = 'alberto.santos@cpr.ku.dk'
from Bio import Medline
import re
import pandas as pd
import numpy as np
from dask import dataframe as dd
import bs4 as bs
import dash_html_components as html
import networkx as nx
from networkx.readwrite import json_graph
import chart_studio.plotly as py
import base64
from xhtml2pdf import pisa
import requests, json
from urllib import request, parse, error
import shutil
import smtplib
from email.message import EmailMessage

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

def send_message_to_slack_webhook(message, message_to):
    webhook_file = "../config/wh.txt"
    with open(webhook_file, 'r') as hf:
        webhook_url = hf.read()
 
    post = {"text": "@{} : {}".format(message_to, message), "username":"albsantosdel", "icon_url": "https://slack.com/img/icons/app-57.png"}
 
    try:
        json_data = json.dumps(post)
        req = request.Request(webhook_url,
                              data=json_data.encode('ascii'),
                              headers={'Content-Type': 'application/json'}) 
        resp = request.urlopen(req)
    except Exception as em:
        print("EXCEPTION: " + str(em))

def send_email(message, subject, message_from, message_to):
    msg = EmailMessage()    
    msg.set_content(message)
    msg['Subject'] = subject
    msg['From'] = message_from
    msg['to'] = message_to

    s = smtplib.SMTP('localhost:1025') 
    s.send_message(msg)
    s.quit()

def compress_directory(name, directory, compression_format='zip'):
    try:
        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.make_archive(name, compression_format, directory)
            shutil.rmtree(directory)
    except Exception as err:
        print("Could not compress file {} in directory {}. Error: {}".format(name, directory, err))

def get_markdown_date(extra_text):
    today = date.today()
    current_date = today.strftime("%B %d, %Y")
    markdown =  html.Div([dcc.Markdown('### *{} {}* ###'.format(extra_text,current_date))],style={'color': '#6a51a3'})
    return markdown

def convert_html_to_dash(el,style = None):
    if type(el) == bs.element.NavigableString:
        return str(el)
    else:
        name = el.name
        style = extract_style(el) if style is None else style
        contents = [convert_html_to_dash(x) for x in el.contents]
        return getattr(html,name.title())(contents,style = style)

def extract_style(el):
    return {k.strip():v.strip() for k,v in [x.split(": ") for x in el.attrs["style"].split(";")]}

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def convert_dash_to_json(dash_object):
    if not hasattr(dash_object, 'to_plotly_json'):
        dash_json = dash_object
    else:
        dash_json = dash_object.to_plotly_json()
        for key in dash_json:
            if isinstance(dash_json[key], dict):
                for element in dash_json[key]:
                    children = dash_json[key][element]
                    ch = {element:[]}
                    if is_jsonable(children) or isinstance(children, np.ndarray): 
                        ch[element] = children
                    elif isinstance(children, dict):
                        ch[element] = {}
                        for c in children:
                            ch[element].update({c:[]})
                            if isinstance(children[c], list):
                                for f in children[c]:
                                    if is_jsonable(f) or isinstance(f, np.ndarray):
                                        ch[element][c].append(f)
                                    else:
                                        ch[element][c].append(convert_dash_to_json(f))
                            else:
                                if is_jsonable(children[c]) or isinstance(children[c], np.ndarray):
                                    ch[element][c] = children[c]
                                else:
                                    ch[element][c] = convert_dash_to_json(children[c])
                    elif isinstance(children, list): 
                        for c in children: 
                            if is_jsonable(c) or isinstance(c, np.ndarray):
                                ch[element].append(c)
                            else: 
                                ch[element].append(convert_dash_to_json(c))
                    else:
                        ch[element] = convert_dash_to_json(children)
                    dash_json[key].update(ch)
    return dash_json 

def neoj_path_to_networkx(paths, key='path'):
    regex = r"\(?(.+)\)\<?\-\>?\[\:(.+)\s\{.*\}\]\<?\-\>?\((.+)\)?"
    nodes = set()
    rels = set()
    for r in paths:
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

def networkx_to_cytoscape(graph):
    cy_graph = json_graph.cytoscape_data(graph)
    cy_nodes = cy_graph['elements']['nodes']
    cy_edges = cy_graph['elements']['edges']
    cy_elements = cy_nodes
    cy_elements.extend(cy_edges)

    return cy_elements


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

def get_image(figure, width, height):
    img = base64.b64encode(py.image.get(figure, width=width, height=height)).decode('utf-8')

    return img
    
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

def getNumberText(num):
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"]
    if len(numbers) > num:
        return numbers[num]
    else:
        return None

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
    except error.HTTPError:
        print("Request to Bio.Entrez failed")

    return abstracts


# Utility function
def convert_html_to_pdf(source_html, output_filename):
    # open output file for writing (truncated binary)
    result_file = open(output_filename, "w+b")

    # convert HTML to PDF
    pisa_status = pisa.CreatePDF(
            source_html,                # the HTML to convert
            dest=result_file)           # file handle to recieve result

    # close output file
    result_file.close()                 # close output file

    # return True on success and False on errors
    return pisa_status.err

def expand_dataframe_cell(data, col, sep):
    '''
        data: pandas dataframe
        col: column/s you need to expand
        sep: separator in col
    '''
    ddata = dd.from_pandas(data, 6)
    ddata = ddata.map_partitions(lambda df: df.drop(col, axis=1).join(df[col].str.split(';', expand=True).stack().reset_index(drop=True, level=1).rename(col))).compute()
    
    return ddata

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
