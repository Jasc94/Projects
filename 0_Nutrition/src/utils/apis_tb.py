from flask import Flask, request, render_template, Response

import numpy as np
import pandas as pd
import joblib
import json

import sys, os

# Helpers
abspath = os.path.abspath
dirname = os.path.dirname
sep = os.sep

# Update sys.path for in-house libraries
folder_ = dirname(abspath(__file__))
sys.path.append(folder_)

# In-house libraries
import folder_tb as fo
import mining_data_tb as md

# Flask object
app = Flask(__name__)

# Data path
environment_path = fo.path_to_folder(2, "data" + sep + "environment")
health_path = fo.path_to_folder(2, "data" + sep + "health")

##################################################### API FUNCTIONS #####################################################
####
@app.route("/")
def home():
    return "Gracias por venir"

####
@app.route("/resources", methods = ["GET"])
def resources():
    resources = pd.read_csv(environment_path + "resources.csv", index_col = 0)
    return resources.to_json()

####
@app.route("/nutrition", methods = ["GET"])
def nutrition():
    nutrition = pd.read_csv(environment_path + "nutritional_values.csv", index_col = 0)
    return nutrition.to_json()

####
@app.route("/health", methods = ["GET"])
def health():
    health = pd.read_csv(environment_path + "7_cleaned_data" + sep + "cleaned_data.csv", index_col = 0)
    return health.to_json()

####
@app.route("/health-variables", methods = ["GET"])
def health_variables():
    health_variables = pd.read_csv(environment_path + "6_variables" + sep + "0_final_variables.csv", index_col = 0)
    return health_variables.to_json()

##################################################### MAIN FUNCTION #####################################################
def main():
    print("--- STARTING PROCESS ---")
    print(__file__)

    settings_path = fo.path_to_folder(2, "src" + sep + "api") + "settings.json"
    print("settings path:\n", settings_path)

    SERVER_RUNNING = md.read_json["SERVER_RUNNING"]
    print("SERVER_RUNNING", SERVER_RUNNING)

    if SERVER_RUNNING:
        DEBUG = md.read_json["DEBUG"]
        HOST = md.read_json["HOST"]
        PORT = md.read_json["PORT"]

        app.run(debug = DEBUG, host = HOST, port = PORT)

    else:
        print("Server settings.json doesn't allow to start server. " + 
            "Please, allow it to run it.")