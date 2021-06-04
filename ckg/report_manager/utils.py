import os
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
from ckg.graphdb_connector import connector
from datetime import datetime
import bs4 as bs
import random
import numpy as np
import chart_studio.plotly as py
import base64
from xhtml2pdf import pisa
import json
from urllib import request
import shutil
import smtplib
from email.message import EmailMessage
from ckg import ckg_utils


def copy_file_to_destination(cfile, destination):
    if os.path.exists(destination):
        shutil.copyfile(cfile, destination)


def send_message_to_slack_webhook(message, message_to, username='albsantosdel'):
    webhook_file = os.path.join(ckg_utils.read_ckg_config(key='ckg_directory'), "config/wh.txt")
    if os.path.exists(webhook_file):
        with open(webhook_file, 'r') as hf:
            webhook_url = hf.read()
        post = {"text": "@{} : {}".format(message_to, message), "username": username, "icon_url": "https://slack.com/img/icons/app-57.png"}
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
            if os.path.exists(directory+'.zip'):
                os.remove(directory+'.zip')
            shutil.make_archive(name, compression_format, directory)
            shutil.rmtree(directory)
        else:
            print("The directory {} failed. Exists?{} Is dir?{}".format(directory, os.path.exists(directory) and os.path.isdir(directory)))
    except Exception as err:
        print("Could not compress file {} in directory {}. Error: {}".format(name, directory, err))


def get_markdown_date(extra_text):
    now = datetime.now()
    current_date = now.strftime("%B %d, %Y (%H:%M:%S)")
    markdown = html.Div([dcc.Markdown('### *{} {}* ###'.format(extra_text, current_date))], style={'color': '#6a51a3'})
    return markdown


def convert_html_to_dash(el, style=None):
    if type(el) == bs.element.NavigableString:
        return str(el)
    else:
        name = el.name
        style = extract_style(el) if style is None else style
        contents = [convert_html_to_dash(x) for x in el.contents]
        return getattr(html, name.title())(contents, style=style)


def extract_style(el):
    return {k.strip(): v.strip() for k, v in [x.split(": ") for x in el.attrs["style"].split(";")]}


def get_image(figure, width, height):
    img = base64.b64encode(py.image.get(figure, width=width, height=height)).decode('utf-8')

    return img


def parse_html(html_snippet):
    html_parsed = bs.BeautifulSoup(html_snippet)

    return html_parsed    


def hex2rgb(color):
    hex = color.lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
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