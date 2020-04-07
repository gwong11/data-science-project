import sqlite3
from sqlite3 import Error
import pandas as pd

class DatabaseIngest:
    '''
    Initialize the database file
    '''
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = None

    '''
    Create a database connection
    '''
    def create_connection(self): 
        try:
            self.conn = sqlite3.connect(self.db_file)
        except Error as e:
            print(e)

    '''
    Create table
    '''
    def create_table(self, table, schema):
        try:
            sql = 'CREATE TABLE IF NOT EXISTS ' + table + ' (' + schema + ');'
            curr = self.conn.cursor()
            curr.execute(sql)
            self.conn.commit()
        except Error as e:
            print(e)

    '''
    Insert record
    '''
    def insert_record(self, table, exist_strategy, df):
        try:
            df.to_sql(table, self.conn, if_exists=exist_strategy, index=False)
        except Error as e:
            print(e)

    '''
    Delete all record
    '''
    def delete_record(self, table):
        try:
            sql = 'DELETE FROM ' + table
            cur = self.conn.cursor()
            cur.execute(sql)
            #cur.execute('vacuum')
            self.conn.commit()
        except Error as e:
            print(e)

    '''
    Drop the table
    '''
    def drop_table(self, table):
        try:
            sql = 'DROP TABLE ' + table
            cur = self.conn.cursor()
            cur.execute(sql)
            #cur.execute('vacuum')
            self.conn.commit()
        except Error as e:
            print(e)

    '''
    Query database
    '''
    def query_record(self, sql_statement):
        try:
            df = pd.read_sql_query(sql_statement, self.conn)
            return df
        except Error as e:
            print(e)

    '''
    Commit transaction
    '''
    def commit(self):
        self.conn.commit()

    '''
    Close database
    '''
    def close(self):
        self.conn.close()
