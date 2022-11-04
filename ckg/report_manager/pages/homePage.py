import dash
import flask
import pandas as pd
from dash import html, dcc, Output, Input

from ckg.report_manager.apps import homepageStats as hpstats

title = "CKG homepage"
subtitle = "Database Stats"
description = ""

dash.register_page(__name__, path='/', title=f"{title} - {subtitle}", description=description)


def layout():
    session_cookie = flask.request.cookies.get('custom-auth-session')
    logged_in = session_cookie is not None

    print(logged_in)

    if logged_in == False:
        return html.Div(["Please ", dcc.Link("login", href="/apps/loginPage"), " to continue"])
    else:
        plots = get_plots()

        homepage_layout = html.Div(children=[
            html.H1(children=title),
            html.H2(children=subtitle),
            html.Div(children=description),

            html.Div(hpstats.quick_numbers_panel()),

            html.Div(plots[0]),
            html.Div(plots[1]),
            html.Div(plots[2])
        ])
        return homepage_layout


def get_plots():
    args = {}
    args['valueCol'] = 'value'
    args['textCol'] = 'size'
    args['y'] = 'index'
    args['x'] = 'number'
    args['orientation'] = 'h'
    args['title'] = ''
    args['x_title'] = ''
    args['y_title'] = ''
    args['height'] = 900
    args['width'] = 900
    dfs = hpstats.get_db_stats_data()
    plots = []
    plots.append(hpstats.plot_store_size_components(dfs, title='DB Store Size', args=args))
    plots.append(hpstats.plot_node_rel_per_label(dfs, focus='nodes', title='Nodes per Label', args=args))
    plots.append(hpstats.plot_node_rel_per_label(dfs, focus='relationships', title='Relationships per Type', args=args))

    return plots


@dash.callback(Output('db-creation-date', 'children'),
               [Input('db_stats_df', 'data')])
def update_db_date(df):
    db_date = "Unknown"
    if 'kernel_monitor' in df:
        kernel = pd.read_json(df['kernel_monitor'], orient='records')
        db_date = kernel['storeCreationDate'][0]

    return html.H3('Store Creation date: {}'.format(db_date))


@dash.callback([Output("db_indicator_14", "children"),
                Output("db_indicator_1", "children"),
                Output("db_indicator_3", "children"),
                Output("db_indicator_2", "children"),
                Output("db_indicator_4", "children"),
                Output("db_indicator_5", "children"),
                Output("db_indicator_6", "children"),
                Output("db_indicator_7", "children"),
                Output("db_indicator_8", "children"),
                Output("db_indicator_9", "children"),
                Output("db_indicator_10", "children"),
                Output("db_indicator_11", "children"),
                Output("db_indicator_12", "children"),
                Output("db_indicator_13", "children"),
                ],
               [Input("db_stats_df", "data")])
def number_panel_update(df):
    print("Update func")
    updates = []
    if 'projects' in df:
        projects = pd.read_json(df['projects'], orient='records')
        if not projects.empty and 'Projects' in projects:
            projects = projects['Projects'][0]
        updates.append(projects)
        if 'meta_stats' in df:
            meta_stats = pd.read_json(df['meta_stats'], orient='records')
            if not meta_stats.empty:
                if 'nodeCount' in meta_stats:
                    ent = meta_stats['nodeCount'][0]
                else:
                    ent = '0'
                updates.append(ent)
                if 'relCount' in meta_stats:
                    rel = meta_stats['relCount'][0]
                else:
                    rel = '0'
                updates.append(rel)
                if 'labelCount' in meta_stats:
                    labels = meta_stats['labelCount'][0]
                else:
                    labels = '0'
                updates.append(labels)
                if 'relTypeCount' in meta_stats:
                    types = meta_stats['relTypeCount'][0]
                else:
                    types = '0'
                updates.append(types)
                if 'propertyKeyCount' in meta_stats:
                    prop = meta_stats['propertyKeyCount'][0]
                else:
                    prop = '0'
                updates.append(prop)

    if 'store_size' in df:
        store_size = pd.read_json(df['store_size'], orient='records')
        if not store_size.empty and 'size' in store_size:
            ent_store = store_size['size'][2]
            rel_store = store_size['size'][4]
            prop_store = store_size['size'][3]
            string_store = store_size['size'][5]
            array_store = store_size['size'][0]
            log_store = store_size['size'][1]
        else:
            ent_store = '0 MB'
            rel_store = '0 MB'
            prop_store = '0 MB'
            string_store = '0 MB'
            array_store = '0 MB'
            log_store = '0 MB'

        updates.extend([ent_store, rel_store, prop_store, string_store, array_store, log_store])

    if 'transactions' in df:
        transactions = pd.read_json(df['transactions'], orient='records')
        if not transactions.empty and 'name' in transactions:
            t_open = transactions.loc[transactions['name'] == 'NumberOfOpenedTransactions', 'value'].iloc[0]
            t_comm = transactions.loc[transactions['name'] == 'NumberOfCommittedTransactions', 'value'].iloc[0]
        else:
            t_open = '0'
            t_comm = '0'

        updates.extend([t_open, t_comm])

    return [dcc.Markdown("**{}**".format(i)) for i in updates]
