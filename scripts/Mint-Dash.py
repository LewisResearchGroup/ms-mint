import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from tkinter import Tk, filedialog

from mint.backend import Mint

mint = Mint()

app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)

app.layout = html.Div(
    [
        html.H1("Mint-Dash"),
        html.Button(id='mzxml'),
        html.Div(id='mzxml-output',
                 children=''),
        dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
    ]
)

@app.callback(
    Output('mzxml', 'children'),
    [Input('mzxml', 'n_clicks')] )
def update_something(n_clicks):
    if n_clicks is None:
        return 'Select mzXML files'
    root = Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    files = filedialog.askopenfilename(multiple=True)
    files = [i  for i in files if i.endswith('.mzxml')]
    if len(files) != 0:
        mint.mzxml_files = files
    root.destroy()
    return '{} mzXML-files selected'.format(len(files))

app.run_server(debug=True, port=9997)
