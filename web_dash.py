import dash
#import dash_core_components as dcc
from dash import dcc
from dash import html
#import dash_html_components as html
import pandas as pd
from plotly_3d import *
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

data = 'test_data.npy'
fig = model_3d(data)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.H1(children="Interactive Bone Visualization"),
    html.P(children="3D Visualization of the selected bone!"),
    dcc.Graph(id='MyGraph'),
    html.Div(id='output-data-upload')
    ])

@app.callback(Output('MyGraph', 'figure'),
            [
                Input('upload-data', 'contents'),
                Input('upload-data', 'filename')
            ])
def update_graph(contents, filename):
    print(contents[0])
    contents = np.load(contents[0])
    print(type(contents))
    print(contents.shape)
    fig = model_3d(contents)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)