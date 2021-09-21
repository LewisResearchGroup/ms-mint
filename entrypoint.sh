#!/usr/bin/env bash
cd flaskapp
flask db upgrade
#flask run --host=127.0.0.1
python serve.py