import dash_html_components as html
import bs4 as bs
import random
from Bio import Entrez
Entrez.email = 'alberto.santos@cpr.ku.dk'
from Bio import Medline
import pandas as pd
from dask import dataframe as dd
import plotly.plotly as py
import base64
from xhtml2pdf import pisa
import requests, json

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
    
    print(link)
    
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

def extract_style(el):
    return {k.strip():v.strip() for k,v in [x.split(": ") for x in el.attrs["style"].split(";")]}

def convert_html_to_dash(el,style = None):
    if type(el) == bs.element.NavigableString:
        return str(el)
    else:
        name = el.name
        style = extract_style(el) if style is None else style
        contents = [convert_html_to_dash(x) for x in el.contents]
        return getattr(html,name.title())(contents,style = style)

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
