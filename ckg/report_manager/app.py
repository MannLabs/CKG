import os

import dash
import flask
import redis

server = flask.Flask('app')
cwd = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(cwd, 'assets')
os.chdir(cwd)
pages_path = "./pages"
application = dash.Dash("app", server=server, assets_folder=assets_path,
                        external_stylesheets=[assets_path + "custom.css"],
                        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                        use_pages=True,
                        pages_folder=pages_path)

r = redis.StrictRedis.from_url('redis://localhost:6379')

with open(os.path.join(assets_path, "app_template.html"), 'r', encoding='utf8') as f:
    template = f.read()

application.index_string = template
application.scripts.config.serve_locally = False
application.config.suppress_callback_exceptions = True

external_js = ["http://code.jquery.com/jquery-3.4.1.min.js"]

for js in external_js:
    application.scripts.append_script({"external_url": js})
