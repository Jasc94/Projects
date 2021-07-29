from flask import Flask, request, render_template, Response

import pandas as pd
import json

import sys, os

# Helpers
abspath = os.path.abspath
dirname = os.path.dirname
sep = os.sep

# Update sys.path for in-house libraries
folder_ = dirname(abspath(__file__))
for i in range(1): folder_ = dirname(folder_)
sys.path.append(folder_)

# In-house libraries
import utils.folder_tb as fo
import utils.apis_tb as ap

##################################################### RUN SERVER #####################################################
if __name__ == "__main__":
    ap.main()