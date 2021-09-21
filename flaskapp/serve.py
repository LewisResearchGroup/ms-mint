from waitress import serve

from flaskapp import app

serve(app, port=5000)