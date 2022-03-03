import os

# import configparser

from pathlib import Path as P
from flask_login.mixins import UserMixin

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine


db = SQLAlchemy()


class ConnectDB:
    def __init__(self, path, app):

        server = app.server

        full_path = P(path) / "data.sqlite"

        self.engine = create_engine(f"sqlite:///{full_path}")

        self.db = db

        # config = configparser.ConfigParser()

        server.config.update(
            SECRET_KEY=os.urandom(12),
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{full_path}",
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )

        app.db = self.db

        app.engine = self.engine

        db.init_app(server)

        self.create_users_table()

    def create_users_table(self):
        Users.metadata.create_all(self.engine)


class Users(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True, nullable=False)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
