#!/usr/bin/env bash

cd flaskapp

export FLASK_APP='flaskapp:create_app'
export SQLALCHEMY_DATABASE_URI='sqlite:////data/mint.db'
flask db upgrade

waitress-serve --port 8000 --call 'flaskapp:create_app'

