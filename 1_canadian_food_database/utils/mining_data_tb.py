import pandas as pd

import sys, os


def data_to_tables(path):
    """It reads all csv files in a folder as dataframes and stores them in a dictionary as values with the file name in lowercase and spaces replaced with "_" as keys,

    Args:
        path (str): Path to folder

    Returns:
        dict: Dictionary with file name as key and dataframe as value
    """
    tables = {}

    for i in os.listdir(path):
        # Getting file name and cleaning it
        table_name = i.lower().replace(" ", "_")[:-4]
        # Saving data as dataframes
        table_data = pd.read_csv(path + i)
        # Saving name(key) and data(value)
        tables[table_name] = table_data

    return tables