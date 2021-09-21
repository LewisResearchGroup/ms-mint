from waitress import serve

from flaskapp import app

serve(app, host='0.0.0.0', port=8000, url_scheme='http')

