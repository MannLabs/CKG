import os
import subprocess
from datetime import datetime
from uuid import uuid4

import dash
import flask
import redis
from dash import html, dcc, Output, Input

import ckg.report_manager.user as user
from ckg import ckg_utils
from ckg.graphdb_builder import builder_utils
from ckg.report_manager import utils
from ckg.report_manager.worker import run_minimal_update_task, \
    run_full_update_task

ckg_config = ckg_utils.read_ckg_config()
log_config = ckg_config['report_manager_log']
logger = builder_utils.setup_logging(log_config, key="app")
config = builder_utils.setup_config('builder')
separator = config["separator"]

flask_server = flask.Flask('app')
cwd = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(cwd, 'assets')
os.chdir(cwd)
pages_path = "./pages"
app = dash.Dash("app", server=flask_server, assets_folder=assets_path,
                external_stylesheets=[assets_path + "custom.css"],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                use_pages=True,
                pages_folder=pages_path)

r = redis.StrictRedis.from_url('redis://localhost:6379')

with open(os.path.join(assets_path, "app_template.html"), 'r', encoding='utf8') as f:
    template = f.read()

app.index_string = template
app.scripts.config.serve_locally = False
app.config.suppress_callback_exceptions = True

external_js = ["http://code.jquery.com/jquery-3.4.1.min.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})

app.layout = html.Div(children=[
    html.Div(id="user-status-header"),
    html.Hr(),

    dcc.Loading(children=[html.Div([dcc.Location(id='url', refresh=False),
                                    html.Div(id='page-content',
                                             style={'padding-top': 10},
                                             className='container-fluid'),
                                    dash.page_container])],
                style={'text-align': 'center',
                       'top': '50%',
                       'left': '50%',
                       'height': '250px'},
                type='cube', color='#2b8cbe'),
])


@dash.callback(
    Output("user-status-header", "children"),
    Input("url", "pathname"),
)
def update_authentication_status(_):
    session_cookie = flask.request.cookies.get('custom-auth-session')
    logged_in = session_cookie is not None
    if logged_in:
        return dcc.Link([html.Form([html.Button('Logout', type='submit')], action='/apps/logout', method='post',
                                   style={'position': 'absolute', 'right': '0px'}, id='logout')],
                        href="/apps/logoutPage")
    return dcc.Link(html.Form([html.Button('Login', type='submit')],
                              style={'position': 'absolute', 'right': '0px'}, id='login'), href="/apps/loginPage")


@flask_server.route('/apps/login', methods=['POST', 'GET'])
def route_login():
    data = flask.request.form
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        flask.abort(401)
    elif not user.User(username).verify_password(password):
        return flask.redirect('/login_error')
    else:
        rep = flask.redirect('/')
        rep.set_cookie('custom-auth-session',
                       username + '_' + datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()))
        return rep


@flask_server.route('/apps/logout', methods=['POST'])
def route_logout():
    # Redirect back to the index and remove the session cookie.
    rep = flask.redirect('/')
    rep.set_cookie('custom-auth-session', '', expires=0)
    return rep


@flask_server.route('/create_user', methods=['POST', 'GET'])
def route_create_user():
    data = flask.request.form
    name = data.get('name')
    surname = data.get('surname')
    affiliation = data.get('affiliation')
    acronym = data.get('acronym')
    email = data.get('email')
    alt_email = data.get('alt_email')
    phone = data.get('phone')
    uname = name[0] + surname
    username = uname

    registered = 'error_exists'
    iter = 1
    while registered == 'error_exists':
        u = user.User(username=username.lower(), name=name, surname=surname, affiliation=affiliation, acronym=acronym,
                      phone=phone, email=email, alternative_email=alt_email)
        registered = u.register()
        if registered is None:
            rep = flask.redirect('/apps/admin?error_new_user={}'.format('Failed Database'))
        elif registered == 'error_exists':
            username = uname + str(iter)
            iter += 1
        elif registered == 'error_email':
            rep = flask.redirect('/apps/admin?error_new_user={}'.format('Email already registered'))
        elif registered == 'error_database':
            rep = flask.redirect('/apps/admin?error_new_user={}'.format('User could not be saved in the database'))
        else:
            rep = flask.redirect('/apps/admin?new_user={}'.format(username))

    return rep


@flask_server.route('/update_minimal', methods=['POST', 'GET'])
def route_minimal_update():
    session_cookie = flask.request.cookies.get('custom-auth-session')
    username = session_cookie.split('_')[0]
    internal_id = datetime.now().strftime('%Y%m-%d%H-%M%S-')
    result = run_minimal_update_task.apply_async(args=[username], task_id='run_minimal_' + session_cookie + internal_id,
                                                 queue='update')

    rep = flask.redirect('/apps/admin/running=minimal')

    return rep


@flask_server.route('/update_full', methods=['POST', 'GET'])
def route_full_update():
    session_cookie = flask.request.cookies.get('custom-auth-session')
    data = flask.request.form
    download = data.get('dwn-radio') == 'true'
    username = session_cookie.split('_')[0]
    internal_id = datetime.now().strftime('%Y%m-%d%H-%M%S-')
    result = run_full_update_task.apply_async(args=[username, download],
                                              task_id='run_full_' + session_cookie + internal_id, queue='update')

    rep = flask.redirect('/apps/admin/running=full')

    return rep


@flask_server.route('/downloads/<value>')
def route_report_url(value):
    uri = os.path.join(ckg_config['downloads_directory'], value + '.zip')
    return flask.send_file(uri, download_name=value + '.zip', as_attachment=True, max_age=-1)


@flask_server.route('/example_files')
def route_example_files_url():
    uri = os.path.join(ckg_config['data_directory'], 'example_files.zip')
    return flask.send_file(uri, download_name='example_files.zip', as_attachment=True, max_age=-1)


@flask_server.route('/apps/templates<value>')
def serve_static(value):
    cwd = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(cwd, 'apps/templates/')
    filename = os.path.join(directory, value)
    url = filename + '.zip'
    if not os.path.isfile(url):
        utils.compress_directory(filename, os.path.join(directory, 'files'), compression_format='zip')

    return flask.send_file(url, download_name=f"{value}.zip", as_attachment=True, max_age=-1)


@flask_server.route('/tmp/<value>')
def route_upload_url(value):
    page_id, project_id = value.split('_')
    directory = ckg_config['tmp_directory']
    filename = os.path.join(directory, 'Uploaded_files_' + project_id)
    url = filename + '.zip'

    return flask.send_file(url, download_name=filename.split('/')[-1] + '.zip', as_attachment=True,
                           max_age=-1)


def main():
    logger.info("Starting CKG App")
    celery_working_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(celery_working_dir)
    queues = [('creation', 1, 'INFO'), ('compute', 3, 'INFO'), ('update', 1, 'INFO')]
    print(type(ckg_config))
    print(ckg_config["log_directory"] + "/celery.log")
    for queue, processes, log_level in queues:
        celery_cmdline = 'celery -A ckg.report_manager.worker worker --loglevel={} --logfile={} --concurrency={} -E -Q {}'.format(
            log_level, ckg_config["log_directory"] + "/celery.log", processes, queue).split(" ")
        logger.info("Ready to call {} ".format(celery_cmdline))
        subprocess.Popen(celery_cmdline)
        logger.info("Done calling {} ".format(celery_cmdline))
    app.run_server(debug=False, host='0.0.0.0')


if __name__ == '__main__':
    main()
