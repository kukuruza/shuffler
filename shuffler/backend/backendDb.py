import logging
import sqlite3
from datetime import datetime
import io
import numpy as np


def _load_db_to_memory(in_db_path):
    # Read database to tempfile
    conn = sqlite3.connect('file:%s?mode=ro' % in_db_path, uri=True)
    tempfile = io.StringIO()
    for line in conn.iterdump():
        tempfile.write('%s\n' % line)
    conn.close()
    tempfile.seek(0)

    # Create a database in memory and import from tempfile
    conn = sqlite3.connect(":memory:")
    conn.cursor().executescript(tempfile.read())
    return conn


def connect(in_db_path, how):
    ''' Connect to database in different ways. '''
    if how == 'read_only':
        conn = sqlite3.connect('file:%s?mode=ro' % in_db_path, uri=True)
    elif how == 'load_to_memory':
        conn = _load_db_to_memory(in_db_path)
    elif how == 'as_write':
        conn = sqlite3.connect(in_db_path)
    return conn


def doesTableExist(cursor, table):
    cursor.execute(
        '''SELECT count(*) FROM sqlite_master
                    WHERE name=? AND type='table';''', (table, ))
    return cursor.fetchone()[0] != 0


def createTableImages(cursor):
    cursor.execute('CREATE TABLE images '
                   '(imagefile TEXT PRIMARY KEY, '
                   'width INTEGER, '
                   'height INTEGER, '
                   'maskfile TEXT, '
                   'timestamp TIMESTAMP, '
                   'name TEXT, '
                   'score REAL '
                   ');')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS images_on_imagefile ON images(imagefile);')


def createTableObjects(cursor):
    cursor.execute('CREATE TABLE objects '
                   '(objectid INTEGER PRIMARY KEY, '
                   'imagefile TEXT, '
                   'x1 REAL, '
                   'y1 REAL, '
                   'width REAL, '
                   'height REAL, '
                   'name TEXT, '
                   'score REAL '
                   ');')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS objects_on_imagefile ON objects(imagefile);'
    )
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS objects_on_objectid ON objects(objectid);')


def createTableProperties(cursor):
    cursor.execute('CREATE TABLE properties '
                   '(id INTEGER PRIMARY KEY, '
                   'objectid INTEGER, '
                   'key TEXT, '
                   'value TEXT '
                   ');')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS properties_on_key ON properties(key);')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS properties_on_key_and_value ON properties(key,value);'
    )
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS properties_on_objectid ON properties(objectid);'
    )


def createTablePolygons(cursor):
    cursor.execute('CREATE TABLE polygons '
                   '(id INTEGER PRIMARY KEY, '
                   'objectid INTEGER, '
                   'x REAL, '
                   'y REAL, '
                   'name TEXT '
                   ');')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS polygons_on_id ON polygons(id);')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS polygons_on_objectid ON polygons(objectid);'
    )


def createTableMatches(cursor):
    cursor.execute('CREATE TABLE matches '
                   '(id INTEGER PRIMARY KEY, '
                   'match INTEGER, '
                   'objectid INTEGER '
                   ');')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS matches_on_match ON matches(match);')


def createDb(conn):
    ''' Creates all the necessary tables and indexes. '''

    cursor = conn.cursor()
    conn.execute('PRAGMA user_version = 5')  # This is version 5.
    createTableImages(cursor)
    createTableObjects(cursor)
    createTableProperties(cursor)
    createTablePolygons(cursor)
    createTableMatches(cursor)


def retireTables(cursor, names=None):
    ''' Changes names of tables to ***_old, and recreates brand-new tables. '''

    if names is None or 'images' in names:
        cursor.execute('ALTER TABLE images RENAME TO images_old')
        createTableImages(cursor)
    if names is None or 'objects' in names:
        cursor.execute('ALTER TABLE objects RENAME TO objects_old')
        createTableObjects(cursor)
    if names is None or 'properties' in names:
        cursor.execute('ALTER TABLE properties RENAME TO properties_old')
        createTableProperties(cursor)
    if names is None or 'matches' in names:
        cursor.execute('ALTER TABLE matches RENAME TO matches_old')
        createTableMatches(cursor)
    if names is None or 'polygons' in names:
        cursor.execute('ALTER TABLE polygons RENAME TO polygons_old')
        createTablePolygons(cursor)


def dropRetiredTables(cursor):
    ''' Drops tables with names ***_old. To be used after retireTables. '''

    if doesTableExist(cursor, 'images_old'):
        cursor.execute('DROP TABLE images_old;')
    if doesTableExist(cursor, 'objects_old'):
        cursor.execute('DROP TABLE objects_old;')
    if doesTableExist(cursor, 'properties_old'):
        cursor.execute('DROP TABLE properties_old;')
    if doesTableExist(cursor, 'matches_old'):
        cursor.execute('DROP TABLE matches_old;')
    if doesTableExist(cursor, 'polygons_old'):
        cursor.execute('DROP TABLE polygons_old;')


def makeTimeString(time):
    ''' Write a time string in Shuffler format.
    Args:      time -- a datetime.datetime object.
    Returns:   string.
    '''
    return datetime.strftime(time, '%Y-%m-%d %H:%M:%S.%f')


def parseTimeString(timestring):
    ''' Parses the Shuffler format.
  Args:      timestring -- a string object
  Returns:   datetime.datetime object
  '''
    return datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S.%f')


def getColumnsInTable(c, table):
    if not doesTableExist(c, table):
        raise IOError('table %s does not exist' % table)
    c.execute('PRAGMA table_info(%s);' % table)
    # Get the first column from the result.
    return [c[1] for c in c.fetchall()]


def objectField(entry, field):
    ''' Convenience function to access by field name. '''

    if field == 'objectid': return entry[0]
    if field == 'imagefile': return entry[1]
    if field == 'x1': return entry[2]
    if field == 'y1': return entry[3]
    if field == 'width': return entry[4]
    if field == 'height': return entry[5]
    if field == 'name': return entry[6]
    if field == 'score': return entry[7]
    if field == 'bbox':
        if None in list(entry[2:6]):
            return None
        else:
            return list(entry[2:6])
    if field == 'roi':
        if None in list(entry[2:6]):
            return None
        else:
            bbox = list(entry[2:6])
            return [bbox[1], bbox[0], bbox[3] + bbox[1], bbox[2] + bbox[0]]
    raise KeyError('No field "%s" in object entry %s' % (field, entry))


def objectFields(entry, fields):
    return [objectField(entry, field) for field in fields]


def setObjectField(entry, field, value):
    ''' Setter for object entry fields. '''

    entry = list(entry)
    if field == 'objectid': entry[0] = value
    elif field == 'imagefile': entry[1] = value
    elif field == 'x1': entry[2] = value
    elif field == 'y1': entry[3] = value
    elif field == 'width': entry[4] = value
    elif field == 'height': entry[5] = value
    elif field == 'name': entry[6] = value
    elif field == 'score': entry[7] = value
    else: raise KeyError('No field "%s" in object entry %s' % (field, entry))
    return tuple(entry)


def objectEntryToDict(entry):
    ''' Convert the tuple returned by sqlite3 SELECT into dict. '''

    return {
        'objectid': entry[0],
        'imagefile': entry[1],
        'x1': entry[2],
        'y1': entry[3],
        'width': entry[4],
        'height': entry[5],
        'name': entry[6],
        'score': entry[7]
    }


def imageField(entry, field):
    ''' Convenience function to access by field name. '''

    if field == 'imagefile': return entry[0]
    if field == 'width': return entry[1]
    if field == 'height': return entry[2]
    if field == 'maskfile': return entry[3]
    if field == 'timestamp': return entry[4]
    if field == 'name': return entry[5]
    if field == 'score': return entry[6]
    raise KeyError('No field "%s" in image entry %s' % (field, entry))


def imageFields(entry, fields):
    return [imageField(entry, field) for field in fields]


def setImageField(entry, field, value):
    ''' Setter for image entry fields. '''

    entry = list(entry)
    if field == 'imagefile': entry[0] = value
    elif field == 'width': entry[1] = value
    elif field == 'height': entry[2] = value
    elif field == 'maskfile': entry[3] = value
    elif field == 'timestamp': entry[4] = value
    elif field == 'name': entry[5] = value
    elif field == 'score': entry[6] = value
    else: raise KeyError('No field "%s" in image entry %s' % (field, entry))
    return tuple(entry)


def polygonField(entry, field):
    ''' Convenience function to access by field name. '''

    if field == 'id': return entry[0]
    if field == 'objectid': return entry[1]
    if field == 'x': return entry[2]
    if field == 'y': return entry[3]
    if field == 'name': return entry[4]
    raise KeyError('No field "%s" in polygon entry %s' % (field, entry))


def polygonFields(entry, fields):
    return [polygonField(entry, field) for field in fields]


def deleteObject(cursor, objectid):
    ''' Delete entries from all tables associated with the object.
  If the object does not exist, raises KeyError.
  '''
    cursor.execute('SELECT COUNT(1) FROM objects WHERE objectid=?;',
                   (objectid, ))
    if cursor.fetchone()[0] == 0:
        raise KeyError(
            'Can not delete objectid %d, as it is not in the database' %
            objectid)
    cursor.execute('DELETE FROM objects WHERE objectid=?;', (objectid, ))
    cursor.execute('DELETE FROM matches WHERE objectid=?;', (objectid, ))
    cursor.execute('DELETE FROM polygons WHERE objectid=?;', (objectid, ))
    cursor.execute('DELETE FROM properties WHERE objectid=?;', (objectid, ))


def deleteImage(cursor, imagefile):
    '''
    Delete entries from all tables associated with the imagefile and all objects
    in this imagefile. If the image does not exist, raises KeyError.
    '''
    cursor.execute('SELECT COUNT(1) FROM images WHERE imagefile=?;',
                   (imagefile, ))
    if cursor.fetchone()[0] == 0:
        raise KeyError(
            'Can not delete imagefile %s, as it is not in the database' %
            imagefile)
    cursor.execute(
        'DELETE FROM matches WHERE objectid IN '
        '(SELECT objectid FROM objects WHERE imagefile=?);', (imagefile, ))
    cursor.execute(
        'DELETE FROM polygons WHERE objectid IN '
        '(SELECT objectid FROM objects WHERE imagefile=?);', (imagefile, ))
    cursor.execute(
        'DELETE FROM properties WHERE objectid IN '
        '(SELECT objectid FROM objects WHERE imagefile=?);', (imagefile, ))
    cursor.execute('DELETE FROM objects WHERE imagefile=?;', (imagefile, ))
    cursor.execute('DELETE FROM images WHERE imagefile=?;', (imagefile, ))


def updateObjectTransform(c, objectid, transform):
    '''
    By convention the "properties" table stores the transformations that
    the object underwent (such as crops, or bbox exansion.)
    Currently only the scale+translation transform is supported.
    The thansform can be used later to infer the original position of an object.
    This function updates the recorded values after a new transform is applied.
    Args:
      c:             cursor
      objectid:      field objectid from "objects" table.
      transform:     a 3x3 float np array [[ky,0,by], [0,kx,bx], [0,0,1]].
    '''
    if transform.shape != (3, 3):
        raise ValueError('Transform must be 3x3 not %s' % str(transform.shape))

    # Get kx.
    c.execute('SELECT id,value FROM properties WHERE objectid=? AND key="kx"',
              (objectid, ))
    entry = c.fetchone()
    kx_id, kx = (entry[0], float(entry[1])) if entry is not None else (None, 1)
    # Get ky.
    c.execute('SELECT id,value FROM properties WHERE objectid=? AND key="ky"',
              (objectid, ))
    entry = c.fetchone()
    ky_id, ky = (entry[0], float(entry[1])) if entry is not None else (None, 1)
    # Get bx.
    c.execute('SELECT id,value FROM properties WHERE objectid=? AND key="bx"',
              (objectid, ))
    entry = c.fetchone()
    bx_id, bx = (entry[0], float(entry[1])) if entry is not None else (None, 0)
    # Get by.
    c.execute('SELECT id,value FROM properties WHERE objectid=? AND key="by"',
              (objectid, ))
    entry = c.fetchone()
    by_id, by = (entry[0], float(entry[1])) if entry is not None else (None, 0)

    transform0 = np.array([[ky, 0., by], [0., kx, bx], [0, 0, 1]])
    logging.debug('Previous transform for objectid %d: \n%s', objectid,
                  str(transform0))

    transform = np.matmul(transform, transform0)
    logging.debug('New transform for objectid %d: \n%s', objectid,
                  str(transform))

    # Update/insert ky.
    if ky_id is not None:
        c.execute('UPDATE properties SET value=? WHERE id=?',
                  (str(transform[0, 0]), ky_id))
    else:
        c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (?,"ky",?)',
            (objectid, str(transform[0, 0])))
    # Update/insert kx.
    if kx_id is not None:
        c.execute('UPDATE properties SET value=? WHERE id=?',
                  (str(transform[1, 1]), kx_id))
    else:
        c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (?,"kx",?)',
            (objectid, str(transform[1, 1])))
    # Update/insert by.
    if by_id is not None:
        c.execute('UPDATE properties SET value=? WHERE id=?',
                  (str(transform[0, 2]), by_id))
    else:
        c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (?,"by",?)',
            (objectid, str(transform[0, 2])))
    # Update/insert bx.
    if bx_id is not None:
        c.execute('UPDATE properties SET value=? WHERE id=?',
                  (str(transform[1, 2]), bx_id))
    else:
        c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (?,"bx",?)',
            (objectid, str(transform[1, 2])))


def upgradeV4toV5(cursor):
    ''' Upgrade the schema to V5, now object coordinates are floating-point. '''

    # Drop indexes in the 'objects' and 'polygons' tables.
    cursor.execute('SELECT name FROM sqlite_master WHERE type == "index" '
                   'AND (name LIKE "objects%" OR name LIKE "polygons%")')
    for index_name, in cursor.fetchall():
        logging.debug('Dropping index: %s', index_name)
        cursor.execute('DROP INDEX "%s"' % index_name)
    # Objects.
    cursor.execute('ALTER TABLE objects RENAME TO objects_old')
    createTableObjects(cursor)
    cursor.execute('INSERT INTO objects SELECT * FROM objects_old;')
    cursor.execute('DROP TABLE objects_old;')
    # Polygons.
    cursor.execute('ALTER TABLE polygons RENAME TO polygons_old')
    createTablePolygons(cursor)
    cursor.execute('INSERT INTO polygons SELECT * FROM polygons_old;')
    cursor.execute('DROP TABLE polygons_old;')
