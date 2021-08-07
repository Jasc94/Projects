import sqlite3

import sys, os


def tables_to_sql(connection, data):
    """[summary]

    Args:
        connection ([type]): [description]
        data ([type]): [description]
    """
    try:
        for key, val in data.items():
            val.to_sql(key, connection, index = False)
        
        print("Tables were correctly loaded")

    except:
        print("Something went wrong")

