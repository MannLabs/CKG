import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import initialApp, projectApp, importsApp


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', style={'padding-top':50})
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname is not None:
        if pathname == '/apps/initial' or pathname == '/':
            return initialApp.layout
        elif pathname.startswith('/apps/project'):
            projectId = pathname.split('/')[-1]
            print(projectId)
            project = projectApp.ProjectApp(projectId, projectId, "", "", layout = [], logo = None, footer = None)

            return project.getLayout()
        elif pathname.startswith('/apps/imports'):
            imports = importsApp.ImportsApp("CKG imports monitoring", "Statistics", "", layout = [], logo = None, footer = None)
            return imports.getLayout()
        else:
            return '404'


if __name__ == '__main__':
    app.run_server(debug=True, port=5000)
