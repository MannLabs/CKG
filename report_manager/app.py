import os
import flask
import redis
import dash
import dash_auth
import dash_cytoscape as cyto

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')


app = dash.Dash('auth', server=server, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = [
    ['hello', 'world']
]

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

r = redis.StrictRedis.from_url('redis://localhost:6379')

with open("app_template.html", 'r') as f:
    template = f.read()


app.index_string = template
app.scripts.config.serve_locally = True
app.config.suppress_callback_exceptions = True
# Dash CSS
external_css = ["https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                "https://codepen.io/chriddyp/pen/bWLwgP.css",
                "https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["http://code.jquery.com/jquery-3.4.1.min.js",
               "https://maxcdn.bootstrapcdn.com/bargsootstrap/4.0.0/js/bootstrap.min.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})
