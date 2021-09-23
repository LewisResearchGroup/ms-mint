import dash
from flask import Flask, request
from flask.helpers import get_root_path
from flask_login import login_required

import dash_bootstrap_components as dbc

from config import BaseConfig

import dash_uploader as du


def create_app():
    app = Flask(__name__, static_url_path='/static')
    app.config.from_object(BaseConfig)

    register_dashapps(app)
    register_extensions(app)
    register_blueprints(app)

    print(app.url_map)

    return app


def register_dashapps(app):
    from ms_mint.app.app import create_app
    from ms_mint.app.app import register_callbacks

    # Meta tags for viewport responsiveness
    meta_viewport = {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}

    dashapp, cache, fsc = create_app(
                            server=app, 
                            meta_tags=[meta_viewport],
                            assets_folder=get_root_path(__name__) + '/assets/',
                            url_base_pathname='/')

    register_callbacks(dashapp, cache, fsc)

    with app.app_context():
        dashapp.title = dashapp.title
        dashapp.layout = dashapp.layout
    _protect_dashviews(dashapp)


def _protect_dashviews(dashapp):
    print( dashapp.server.view_functions )
    for view_func in dashapp.server.view_functions:
        print('F', view_func)
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

