#!/usr/bin/env bash

cd flaskapp

export FLASK_APP='flaskapp:create_app'

flask db upgrade

waitress-serve --port 8080 --call 'flaskapp:create_app'

