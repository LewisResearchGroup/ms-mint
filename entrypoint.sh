#!/usr/bin/env bash
cd flaskapp
flask db upgrade
flask run --host=0.0.0.0
