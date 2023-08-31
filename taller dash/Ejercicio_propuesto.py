import dash
from dash import dcc  # dash core components
from dash import html # dash html components
import plotly.express as px
import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

ventas=pd.DataFrame({"ventas":[6,3,4,5,5,3,2,5,6,5,32,1,1,34,5,5,5,2,2]},index=["A","B","C","A","B","C","A","B","C","A","B","C","A","B","C","A","B","C","B"])
mediaventas=ventas.groupby(level=0).mean()

fig=px.bar(mediaventas)


app.layout = html.Div(children=[
    html.H1(children='Total ventas de productos A.B.C'),

    html.Div(children='''
        GRAFICA DE LA MEDIA DE PRODUCTOS
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    html.Div(children='''
        EN ESTE GRAFICO SE OBSERVA LA MEDIA DE LOS TRES PRODUCTOS A,B,C DADO QUE HICIMOS UNA AGRUPACION DE LA TOTALIDAD DE LOS DATOS
    '''),
    html.Div(
        className="Columnas",
        children=[
            html.Ul(id='my-list', children=[html.Li(i) for i in mediaventas.columns])
        ],
    )
    ]
)

if __name__ == '__main__':
    app.run_s