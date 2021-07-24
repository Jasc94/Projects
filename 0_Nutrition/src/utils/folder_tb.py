import sys, os

# Helpers
abspath = os.path.abspath
dirname = os.path.dirname
sep = os.sep

def path_to_folder(up, folder = ""):
    '''
    Function that calculates the path to a folder

    args :
    up -> how many levels you want to go up
    folder -> once you've gone up "up" levels, the folder you want to open
    '''

    # I start the way up
    path_up = dirname(abspath(__file__))

    # Loop "up" times to reach the main folder
    for i in range(up): path_up = dirname(path_up)

    # go down to the folder I want to pull the data from
    if folder:
        path_down = path_up + sep + folder + sep
    else:
        path_down = path_up + sep

    # return the path
    return path_down