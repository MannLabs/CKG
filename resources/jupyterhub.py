#------------------------------------------------------------------------------
# Configuration file for jupyterhub.
#------------------------------------------------------------------------------

import os
os.umask(0o002)

# set of users who can administer the Hub itself 
c.Authenticator.admin_users = {'adminhub'}
c.Authenticator.whitelist = {'ckguser'}

# set the Clinical Knowledge Graph logo
c.JupyterHub.logo_file = '/CKG/src/report_manager/assets/logo_small.jpg'

## The public facing port of the proxy.
c.JupyterHub.port = 8090
c.Spawner.notebook_dir = '~/notebooks'
c.Spawner.debug = True

#  Supports Linux and BSD variants only.
c.LocalAuthenticator.create_system_users = True

## The command to use for creating users as a list of strings
c.Authenticator.add_user_cmd = ['adduser', '--force-badname', '-q', '--gecos', '""', '--disabled-password']

#Use Google Authenticator
# from oauthenticator.google import GoogleOAuthenticator
# c.JupyterHub.authenticator_class = GoogleOAuthenticator
# c.GoogleOAuthenticator.oauth_callback_url = 'http://example.com/hub/oauth_callback'
# c.GoogleOAuthenticator.client_id = '635823090211-nhef5sl5sqdbq469k4t0l5d14ur7jc8j.apps.googleusercontent.com'
# c.GoogleOAuthenticator.client_secret = 'HA0PdjijSSVog4FUd6nbG9bT'

#Start Jupyterhub as JupyterLAB
#c.Spawner.default_url = '/lab'
