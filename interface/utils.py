import sqlite3
import logging


def openConnection(db_file, mode='r', copy_to_memory=True):
    '''
    Open a sqlite3 database connection.
      db_file:           Path to an sqlite3 database
      mode:              Can be 'r' or 'w'. If 'r', forbid writing.
      copy_to_memory:    A parameter only for read-only mode.
                         If yes, load the whole database to memory.
                         Otherwise, open as file:%s?mode=ro.
    '''
    if mode not in ['w', 'r']:
        raise ValueError('"mode" must be "w" or "r".')
    elif mode == 'w':
        conn = sqlite3.connect(db_file)
    elif copy_to_memory:
        conn = sqlite3.connect(':memory:')  # create a memory database
        disk_conn = sqlite3.connect(db_file)
        query = ''.join(line for line in disk_conn.iterdump())
        conn.executescript(query)
    else:
        try:
            conn = sqlite3.connect('file:%s?mode=ro' % db_file, uri=True)
        except TypeError:
            raise TypeError(
                'This Python version does not support connecting to SQLite by uri'
            )
    return conn
