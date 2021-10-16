 
import dash
import dash_html_components
import dash_core_components as dcc

import dash_html_components as html

print('*'*88)
print('dash:', dash.__version__, dash.__file__)
print('dash_html_components', dash_html_components.__version__, dash_html_components.__file__)


app = dash.Dash(__name__)

app.layout = html.Div('Hello World')

app.run_server(debug=True, 
               dev_tools_hot_reload=False,
               dev_tools_hot_reload_interval=3000,
               dev_tools_hot_reload_max_retry=30)
