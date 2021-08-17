import sqlite3

import sys, os


def tables_to_sql(connection, data):
    """It creates tables in a database using the dictionary key as table name and dictionary value as data.

    Args:
        connection (object): sql connection object
        data (dict): Dictionary with name of the files as keys and table data as values. 
    """
    try:
        for key, val in data.items():
            val.to_sql(key, connection, index = False)
        
        print("Tables were correctly loaded")

    except:
        print("Something went wrong")

