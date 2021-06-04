import os
import flask
import redis
import dash
from ckg import ckg_utils

server = flask.Flask('app')
cwd = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(cwd, 'assets')
app = dash.Dash('app', server=server, assets_folder=assets_path, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])


r = redis.StrictRedis.from_url('redis://localhost:6379')

with open(os.path.join(assets_path, "app_template.html"), 'r', encoding='utf8') as f:
    template = f.read()


app.index_string = template
app.scripts.config.serve_locally = False
app.config.suppress_callback_exceptions = True

external_js = ["http://code.jquery.com/jquery-3.4.1.min.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})

