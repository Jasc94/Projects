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
import utils.dashboard_tb as da

if __name__ == "__main__":
    da.main()