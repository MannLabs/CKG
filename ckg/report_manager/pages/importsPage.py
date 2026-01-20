import os

import dash
import flask
from dash import html, dcc

from ckg import ckg_utils
from ckg.analytics_core.viz import viz
from ckg.report_manager.apps import imports

title = "CKG imports monitoring"
subtitle = "Statistics"
description = ""

dash.register_page(__name__, path='/apps/imports', title=f"{title} - {subtitle}", description=description)


def layout():
    session_cookie = flask.request.cookies.get('custom-auth-session')
    logged_in = session_cookie is not None
    if logged_in == False:
        return html.Div(["Please ", dcc.Link("login", href="/apps/loginPage"), " to continue"])

    plots = get_plots()
    imports_layout = html.Div(children=[
        html.H1(children=title),
        html.H2(children=subtitle),
        html.Div(children=description),
        html.Div(children=[html.Div(plot) for plot in plots]),
    ])
    return imports_layout


def get_plots():
    plots = []
    stats_dir = ckg_utils.read_ckg_config(key='stats_directory')
    stats_file = os.path.join(stats_dir, "stats.hdf")
    if os.path.exists(stats_file):
        stats_df = imports.get_stats_data(stats_file, n=3)
        plots.append(imports.plot_total_number_imported(stats_df, 'Number of imported entities and relationships'))
        plots.append(imports.plot_total_numbers_per_date(stats_df, 'Imported entities vs relationships'))
        plots.append(
            imports.plot_databases_numbers_per_date(stats_df, 'Full imports: entities/relationships per database',
                                                    key='full', dropdown=True, dropdown_options='dates'))
        plots.append(
            imports.plot_databases_numbers_per_date(stats_df, 'Partial imports: entities/relationships per database',
                                                    key='partial', dropdown=True, dropdown_options='dates'))
        plots.append(
            imports.plot_import_numbers_per_database(stats_df, 'Full imports: Breakdown entities/relationships',
                                                     key='full',
                                                     subplot_titles=(
                                                         'Entities imported', 'Relationships imported', 'File size',
                                                         'File size'), colors=True, plots_1='entities',
                                                     plots_2='relationships',
                                                     dropdown=True, dropdown_options='databases'))
        plots.append(
            imports.plot_import_numbers_per_database(stats_df, 'Partial imports: Breakdown entities/relationships',
                                                     key='partial', subplot_titles=(
                    'Entities imported', 'Relationships imported', 'File size', 'File size'), colors=True,
                                                     plots_1='entities',
                                                     plots_2='relationships', dropdown=True,
                                                     dropdown_options='databases'))
    else:
        plots.append(viz.get_markdown(text="# There are no statistics about recent imports."))
    return plots
