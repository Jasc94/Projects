import sys, os

# Helpers
abspath = os.path.abspath
dirname = os.path.dirname
sep = os.sep

def path_to_folder(up, folder = ""):
    """It that calculates the path to a folder

    Args:
        up (int): Folders that you want to go up
        folder (str, optional): Once you've gone up "up" folders, the folder you want to open. Defaults to "".

    Returns:
        str: New path
    """

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