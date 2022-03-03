from waitress import serve

from flaskapp import create_app

serve(create_app, host="127.0.0.1", port=8080, url_scheme="http")
