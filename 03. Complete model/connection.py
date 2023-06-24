import sqlite3
from sqlite3.dbapi2 import Error


def sql_connection():
    try: 
        DATABASE = 'ssd_database.db'
        con = sqlite3.connect(DATABASE)
        cursor = con.cursor()
        return cursor,con
    except Error:
        print(Error)