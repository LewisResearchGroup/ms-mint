import plotly.graph_objects as go
import plotly.io as pio


def set_template():
    pio.templates["draft"] = go.layout.Template(
        layout=dict(font={"size": 10}),
    )

    pio.templates.default = "draft"


set_template()
