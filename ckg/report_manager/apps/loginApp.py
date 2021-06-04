from ckg.report_manager.apps import basicApp
import dash_core_components as dcc
import dash_html_components as html


class LoginApp(basicApp.BasicApp):
    """
    Defines the login App
    Enables user to access the reports
    """
    def __init__(self, title, subtitle, description, layout=[], logo=None, footer=None):
        self.pageType = "loginPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        """
        Builds page with the basic layout from *basicApp.py* and adds the login form.
        """
        self.add_basic_layout()
        login_form = html.Div([html.Form([
                        dcc.Input(placeholder='username', name='username', type='text'),
                        dcc.Input(placeholder='password', name='password', type='password'),
                        html.Button('Login', type='submit')], action='/apps/login', method='post')])
        self.add_to_layout(login_form)
