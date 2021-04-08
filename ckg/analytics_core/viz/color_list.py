"""Code for handling color names and RGB codes.

This module is part of Swampy, and used in Think Python and
Think Complexity, by Allen Downey.

http://greenteapress.com

Copyright 2013 Allen B. Downey.
Distributed under the GNU General Public License at gnu.org/licenses/gpl.html.
"""

import re

# the following is the contents of /etc/X11/rgb.txt

COLORS = """
141 211 199		turquoise
31   120 180		blue
139  69  19		saddlebrown
177  89  40		brown
51 160   44		green
255 237   111		yellow
173 255  47		greenyellow
255   0   0		red
255 255 255		white
0   0   0		black
255 192 203		pink
255   0 255		magenta
160  32 240		purple
210 180 140		tan
250 128 114		salmon
166 206 227		cyan
25  25 112		midnightblue
224 255 255		lightcyan
153 153 153 		grey60
144 238 144		lightgreen
255 255 224		lightyellow
65 105 225		royalblue
139   0   0		darkred
0 100   0		darkgreen
0 206 209		darkturquoise
169 169 169		darkgrey
255 165   0		orange
255 140   0		darkorange
135 206 235		skyblue
70 130 180		steelblue
175 238 238		paleturquoise
238 130 238		violet
85 107  47		darkolivegreen
139   0 139		darkmagenta
190 190 190		gray
190 190 190		grey
"""

def make_color_dict(colors=COLORS):
    """Returns a dictionary that maps color names to RGB strings.

    The format of RGB strings is '#RRGGBB'.
    """
    # regular expressions to match numbers and color names
    number = r'(\d+)'
    space = r'[ \t]*'
    name = r'([ \w]+)'
    pattern = space + (number + space) * 3 + name
    prog = re.compile(pattern)

    # read the file
    d = dict()
    for line in colors.split('\n'):
        ro = prog.match(line)
        if ro:
            r, g, b, name = ro.groups()
            rgb = '#%02x%02x%02x' % (int(r), int(g), int(b))
            d[name] = rgb

    return d


def read_colors():
    """Returns color information in two data structures.

    The format of RGB strings is '#RRGGBB'.

    color_dict: map from color name to RGB string
    rgbs: list of (rgb, names) pairs, where rgb is an RGB code and \
            names is a sorted list of color names
    """
    color_dict = make_color_dict()
    rgbs = invert_dict(color_dict).items()

    return color_dict, rgbs


def invert_dict(d):
    """Returns a dictionary that maps from values to lists of keys.

    d: dict

    returns: dict
    """
    inv = dict()
    for key in d:
        val = d[key]
        if val not in inv:
            inv[val] = [key]
        else:
            inv[val].append(key)
    return inv


if __name__ == '__main__':
    color_dict = make_color_dict()
    for name, rgb in color_dict.items():
        print(name, rgb)

    color_dict, rgbs = read_colors()
    for name, rgb in color_dict.items():
        print(name, rgb)

    for rgb, names in rgbs:
        print(rgb, names)
