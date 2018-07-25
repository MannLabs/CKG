import dash_html_components as html
import bs4 as bs

def parse_html(html_snippet):
    html_parsed = bs.BeautifulSoup(html_snippet)
    print(html_parsed)
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
