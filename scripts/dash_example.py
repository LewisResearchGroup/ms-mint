import plotly_express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

gapminder = px.data.gapminder()

dimensions = []

app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)

app.layout = html.Div(
    [
        html.H1("Gapminder data"),
        dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"},
        figure=px.scatter(gapminder, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
                      size="pop", color="continent", hover_name="country",
                      log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])),
    ]
)


app.run_server(debug=True)