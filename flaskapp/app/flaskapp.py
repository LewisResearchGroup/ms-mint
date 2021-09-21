import dash
from flask import Flask
from flask.helpers import get_root_path
from flask_login import login_required

import dash_bootstrap_components as dbc

from config import BaseConfig


def create_app():
    app = Flask(__name__, static_url_path='/static')
    app.config.from_object(BaseConfig)
    
    register_dashapps(app)
    register_extensions(app)
    register_blueprints(app)

    return app


def register_dashapps(app):
    from ms_mint.app.app import app as dashapp, register_callbacks

    # Meta tags for viewport responsiveness
    meta_viewport = {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}

    dashapp1 = dash.Dash(__name__,
                         server=app,
                         external_stylesheets=[
                            dbc.themes.MINTY,
                            "https://codepen.io/chriddyp/pen/bWLwgP.css"
                         ],
                         url_base_pathname='/',
                         assets_folder=get_root_path(__name__) + '/assets/',
                         meta_tags=[meta_viewport])

    with app.app_context():
        dashapp1.title = 'MINT Server'
        dashapp1.layout = dashapp.layout
        register_callbacks(dashapp1)

    _protect_dashviews(dashapp1)


def _protect_dashviews(dashapp):
    for view_func in dashapp.server.view_functions:
        if view_func.startswith(dashapp.config.url_base_pathname):
            dashapp.server.view_functions[view_func] = login_required(
                dashapp.server.view_functions[view_func])


def register_extensions(server):
    from app.extensions import db
    from app.extensions import login
    from app.extensions import migrate

    db.init_app(server)
    login.init_app(server)
    login.login_view = 'main.login'
    migrate.init_app(server, db)


def register_blueprints(server):
    from app.webapp import server_bp

    server.register_blueprint(server_bp)