import os
import flask
import redis
import dash

server = flask.Flask('app')
app = dash.Dash('app', server=server, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])


r = redis.StrictRedis.from_url('redis://localhost:6379')

cwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cwd, "assets/app_template.html"), 'r', encoding='utf8') as f:
    template = f.read()


app.index_string = template
app.scripts.config.serve_locally = False
app.config.suppress_callback_exceptions = True

external_js = ["http://code.jquery.com/jquery-3.4.1.min.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})

