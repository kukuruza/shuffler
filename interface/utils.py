import sqlite3


def openConnection(db_file, copy_to_memory):

  if copy_to_memory:
    conn = sqlite3.connect(':memory:') # create a memory database
    disk_conn = sqlite3.connect(db_file)
    query = ''.join(line for line in disk_conn.iterdump())
    conn.executescript(query)
  else:
    try:
      conn = sqlite3.connect('file:%s?mode=ro' % db_file, uri=True)
    except TypeError:
      logging.debug('This Python version does not support connecting to SQLite by uri, '
                    'will connect in regular mode (without readonly.)')
      conn = sqlite3.connect(db_file)

  return conn