import os
import subprocess
from datetime import datetime
from uuid import uuid4

import dash
import flask
from dash import html, dcc, Output, Input

import ckg.report_manager.user as user
from ckg import ckg_utils
from ckg.graphdb_builder import builder_utils
from ckg.report_manager import utils
from ckg.report_manager.app import server as application
from ckg.report_manager.worker import run_minimal_update_task, \
    run_full_update_task

try:
    ckg_config = ckg_utils.read_ckg_config()
    log_config = ckg_config['report_manager_log']
    logger = builder_utils.setup_logging(log_config, key="app")
    config = builder_utils.setup_config('builder')
    separator = config["separator"]
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))


def main():
    logger.info("Starting CKG App")
    celery_working_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(celery_working_dir)
    queues = [('creation', 1, 'INFO'), ('compute', 3, 'INFO'), ('update', 1, 'INFO')]
    for queue, processes, log_level in queues:
        celery_cmdline = 'celery -A ckg.report_manager.worker worker --loglevel={} --concurrency={} -E -Q {}'.format(
            log_level, processes, queue).split(" ")
        logger.info("Ready to call {} ".format(celery_cmdline))
        subprocess.Popen(celery_cmdline)
        logger.info("Done calling {} ".format(celery_cmdline))
    application.run_server(debug=False, host='0.0.0.0')


application.layout = html.Div(children=[

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
                        href="apps/logoutPage")
    return dcc.Link(html.Form([html.Button('Login', type='submit')],
                              style={'position': 'absolute', 'right': '0px'}, id='login'), href="/apps/loginPage")


@application.route('/apps/login', methods=['POST', 'GET'])
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


@application.route('/apps/logout', methods=['POST'])
def route_logout():
    # Redirect back to the index and remove the session cookie.
    rep = flask.redirect('/')
    rep.set_cookie('custom-auth-session', '', expires=0)
    return rep


@application.route('/create_user', methods=['POST', 'GET'])
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


@application.route('/update_minimal', methods=['POST', 'GET'])
def route_minimal_update():
    session_cookie = flask.request.cookies.get('custom-auth-session')
    username = session_cookie.split('_')[0]
    internal_id = datetime.now().strftime('%Y%m-%d%H-%M%S-')
    result = run_minimal_update_task.apply_async(args=[username], task_id='run_minimal_' + session_cookie + internal_id,
                                                 queue='update')

    rep = flask.redirect('/dashs/admin?running=minimal')

    return rep


@application.route('/update_full', methods=['POST', 'GET'])
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


@application.route('/downloads/<value>')
def route_report_url(value):
    uri = os.path.join(ckg_config['downloads_directory'], value + '.zip')
    return flask.send_file(uri, download_name=value + '.zip', as_attachment=True, max_age=-1)


@application.route('/example_files')
def route_example_files_url():
    uri = os.path.join(ckg_config['data_directory'], 'example_files.zip')
    return flask.send_file(uri, download_name='example_files.zip', as_attachment=True, max_age=-1)


@application.route('/apps/templates<value>')
def serve_static(value):
    cwd = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(cwd, 'apps/templates/')
    filename = os.path.join(directory, value)
    url = filename + '.zip'
    if not os.path.isfile(url):
        utils.compress_directory(filename, os.path.join(directory, 'files'), compression_format='zip')

    return flask.send_file(url, download_name=f"{value}.zip", as_attachment=True, max_age=-1)


@application.route('/tmp/<value>')
def route_upload_url(value):
    page_id, project_id = value.split('_')
    directory = ckg_config['tmp_directory']
    filename = os.path.join(directory, 'Uploaded_files_' + project_id)
    url = filename + '.zip'

    return flask.send_file(url, download_name=filename.split('/')[-1] + '.zip', as_attachment=True,
                           max_age=-1)


if __name__ == '__main__':
    main()
