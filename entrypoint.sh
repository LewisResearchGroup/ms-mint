#!/usr/bin/env bash
export MINT_DATA_DIR=/data/MINT
export DATABASE_URL=sqlite:///${MINT_DATA_DIR}/app.db
cd flaskapp
flask db upgrade
waitress-serve --port 8000 --call 'flaskapp:create_app'

