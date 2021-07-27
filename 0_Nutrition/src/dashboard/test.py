import pandas as pd

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
import utils.mining_data_tb as md
import utils.visualization_tb as vi

print(fo.path_to_folder(2, "models"))