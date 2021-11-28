from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, current_user

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from sqlalchemy import Table

from .database import Users


_layout = html.Div(
    [
        html.Div(id="page-content", className="content"),
        dcc.Location(id="url", refresh=False),
    ]
)


create = html.Div(
    [
        html.H1("Create User Account"),
        dcc.Location(id="create_user", refresh=True),
        dcc.Input(id="username", type="text", placeholder="user name", maxLength=15),
        dcc.Input(id="password", type="password", placeholder="password"),
        dcc.Input(id="email", type="email", placeholder="email", maxLength=50),
        html.Button("Create User", id="submit-val", n_clicks=0),
        html.Div(id="container-button-basic"),
    ]
)  # end div


login = html.Div(
    [
        dcc.Location(id="url_login", refresh=True),
        html.H2("""Please log in to continue:""", id="h1"),
        dcc.Input(placeholder="Enter your username", type="text", id="uname-box"),
        dcc.Input(placeholder="Enter your password", type="password", id="pwd-box"),
        html.Button(children="Login", n_clicks=0, type="submit", id="login-button"),
        html.Div(children="", id="output-state"),
    ]
)  # end div


success = html.Div(
    [
        dcc.Location(id="url_login_success", refresh=True),
        html.Div(
            [
                html.Br(),
                html.Button(
                    id="back-button",
                    children="Logout",
                    n_clicks=0,
                    style={"float": "right"},
                ),
            ]
        ),
    ]
)

failed = html.Div(
    [
        dcc.Location(id="url_login_df", refresh=True),
        html.Div(
            [
                html.H2("Log in Failed. Please try again."),
                html.Br(),
                html.Div([login]),
                html.Br(),
                html.Button(id="back-button", children="Logout", n_clicks=0),
            ]
        ),  # end div
    ]
)  # end div


logout = html.Div(
    [
        dcc.Location(id="logout", refresh=True),
        html.Br(),
        html.Div(html.H2("You have been logged out - Please login")),
        html.Br(),
        html.Div([login]),
        html.Button(id="back-button", children="Logout", n_clicks=0),
    ]
)  # end div


def callbacks(app, fsc=None, cache=None):
    @app.callback(
        Output("container-button-basic", "children"),
        Input("submit-val", "n_clicks"),
        State("username", "value"),
        State("password", "value"),
        State("email", "value"),
    )
    def insert_users(n_clicks, un, pw, em):
        hashed_password = generate_password_hash(pw, method="sha256")
        if un is not None and pw is not None and em is not None:
            Users_tbl = Table("users", Users.metadata)
            ins = Users_tbl.insert().values(
                username=un,
                password=hashed_password,
                email=em,
            )
            with app.engine.begin() as conn:
                conn.execute(ins)
            return [login]
        else:
            return [
                html.Div(
                    [
                        html.H2("Already have a user account?"),
                        dcc.Link("Click here to Log In", href="/login"),
                    ]
                )
            ]

    @app.callback(
        Output("url_login_success", "pathname"), [Input("back-button", "n_clicks")]
    )
    def logout_dashboard(n_clicks):
        if n_clicks > 0:
            return "/"

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def display_page(pathname):
        if pathname == "/":
            return create
        elif pathname == "/login":
            return login
        elif pathname == "/success":
            if current_user.is_authenticated:
                return success
            else:
                return failed
        elif pathname == "/data":
            if current_user.is_authenticated:
                return None
        elif pathname == "/logout":
            if current_user.is_authenticated:
                logout_user()
                return logout
            else:
                return logout
        else:
            return "404"

    @app.callback(
        Output("url_login", "pathname"),
        Input("login-button", "n_clicks"),
        State("uname-box", "value"),
        State("pwd-box", "value"),
    )
    def successful(n_clicks, input1, input2):
        user = Users.query.filter_by(username=input1).first()
        if user:
            if check_password_hash(user.password, input2):
                login_user(user)
                return "/success"
            else:
                pass
        else:
            pass

    @app.callback(
        Output("output-state", "children"),
        Input("login-button", "n_clicks"),
        State("uname-box", "value"),
        State("pwd-box", "value"),
    )
    def update_output(n_clicks, input1, input2):
        if n_clicks > 0:

            user = Users.query.filter_by(username=input1).first()

            if user:
                if check_password_hash(user.password, input2):
                    return ""
                else:
                    return "Incorrect username or password"
            else:
                return "Incorrect username or password"
        else:
            return ""
