import pandas as pd

import sys, os


def data_to_tables(path):
    """[summary]

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
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