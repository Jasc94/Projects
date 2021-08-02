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
    """It returns the resources data as json
    """
    # Read resources csv as dataframe
    resources = pd.read_csv(environment_path + "resources.csv", index_col = 0)
    # Return it as json
    return resources.to_json()

####
@app.route("/nutrition", methods = ["GET"])
def nutrition():
    """It returns the nutrition data as json
    """
    # Read resources csv as dataframe
    nutrition = pd.read_csv(environment_path + "nutritional_values.csv", index_col = 0)
    # Return it as json
    return nutrition.to_json()

####
@app.route("/health", methods = ["GET"])
def health():
    """It returns the health data as json
    """
    # Read resources csv as dataframe
    health = pd.read_csv(health_path + "7_cleaned_data" + sep + "cleaned_data.csv", index_col = 0)
    # Return it as json
    return health.to_json()

####
@app.route("/health-variables", methods = ["GET"])
def health_variables():
    """It returns the health variables data as json
    """
    # Read resources csv as dataframe
    health_variables = pd.read_csv(health_path + "6_variables" + sep + "0_final_variables.csv", index_col = 0)
    # Return it as json
    return health_variables.to_json()

##################################################### MAIN FUNCTION #####################################################
def main():
    """It runs the flask API
    """
    print("--- STARTING PROCESS ---")
    print(__file__)

    # Path to server information
    settings_path = fo.path_to_folder(2, "src" + sep + "api") + "settings.json"
    print("settings path:\n", settings_path)

    # Save server settings as variable
    read_json = md.read_json(settings_path)

    SERVER_RUNNING = read_json["SERVER_RUNNING"]        # True or False
    print("SERVER_RUNNING", SERVER_RUNNING)

    if SERVER_RUNNING:
        DEBUG = read_json["DEBUG"]
        HOST = read_json["HOST"]
        PORT = read_json["PORT"]

        # Run the API with the given settings
        app.run(debug = DEBUG, host = HOST, port = PORT)

    # Error message
    else:
        print("Server settings.json doesn't allow to start server. " + 
            "Please, allow it to run it.")