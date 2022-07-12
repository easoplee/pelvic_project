from distutils.command.upload import upload
from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from backend_test import predict
import torch
import torchvision.models
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import os
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from config import *

import io
from PIL import Image
from base64 import b64encode
import dash
from dash import dcc
from dash import html
import pandas as pd
from plotly_3d import *
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import plotly.express as px


# define a variable to hold you app
server = Flask(__name__)

external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]
app = dash.Dash(__name__, external_scripts=external_script, requests_pathname_prefix="/app/")

app.scripts.config.serve_locally = True

# fig = "empty"

# define your resource endpoints
@server.route('/', methods=['GET', 'POST'])
def index_page():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        filestr = np.load(uploaded_file.stream) #depth x width x height

        pre = predict(filestr) #mask prediction

        figure = model_3d(pre)

        global fig
        fig = figure

        # app.layout = html.Div([
        #     html.Div([
        #         children = 
        #             html.H1(children="Interactive Bone Visualization", className=" py-3 text-5xl font-bold text-gray-800"),
        #             html.P(children="3D Visualization of the selected bone!")],
        #             html.Div([children=dcc.Graph(figure=fig, style={"width": "100%","height": "100%"}), className="shadow-xl w-full border-3 rounded-sm")], style={"width": "100%","height": "100%",}
        # )],
        # style={
        #                     "width": "68%",
        #                     "height": "800px",
        #                     "display": "inline-block",
        #                     "border": "3px #5c5c5c solid",
        #                     "padding-top": "5px",
        #                     "padding-left": "1px",
        #                     "overflow": "hidden",
        #                 }]
        # )
        app.layout = html.Div(
            [html.H1("Interactive Bone Visualization", className=" py-3 text-5xl font-bold text-gray-800", style={"text-align": "center"}),
            dcc.Graph(figure=fig, className=" shadow-xl py-4 px-24 text-5xl bg-[#1d3557] text-white  font-bold text-gray-800", style={"margin-left":"15%", "margin-right":"15%"})]
        )
        return redirect('/app')

        # im = Image.fromarray(np.uint8(pre))
        # im = im.convert("L")
        # byte_io = io.BytesIO()
        # im.save(byte_io, "JPEG")
        # encoded_img_data = b64encode(byte_io.getvalue())

        # if uploaded_file.filename != '':
        #     uploaded_file.save(uploaded_file.filename)
    else:
        return render_template("home.html")

application = DispatcherMiddleware(
    server,
    {"/app": app.server},
)

# server the app when this file is run
if __name__ == '__main__':
    run_simple("localhost", 8050, application)