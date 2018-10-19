import os, os.path as op
import logging
import sqlite3
from datetime import datetime


def doesTableExist (cursor, table):
  cursor.execute('''SELECT count(*) FROM sqlite_master 
                    WHERE name=? AND type='table';''', (table,))
  return cursor.fetchone()[0] != 0


def isColumnInTable (cursor, table, column):
  if not doesTableExist(cursor, table):
    raise IOError ('table %s does not exist' % table)
  cursor.execute('PRAGMA table_info(%s)' % table)
  return column in [x[1] for x in cursor.fetchall()]


def createTableImages (cursor):
  cursor.execute('CREATE TABLE images '
                '(imagefile TEXT PRIMARY KEY, '
                 'width INTEGER, '
                 'height INTEGER, '
                 'maskfile TEXT, '
                 'timestamp TIMESTAMP, '
                 'name TEXT, '
                 'score REAL '
                 ');')
  cursor.execute('CREATE INDEX images_on_imagefile ON images(imagefile);')


def createTableObjects (cursor):
  cursor.execute('CREATE TABLE objects '
                '(objectid INTEGER PRIMARY KEY, '
                 'imagefile TEXT, '
                 'x1 INTEGER, '
                 'y1 INTEGER, '
                 'width INTEGER, ' 
                 'height INTEGER, '
                 'name TEXT, '
                 'score REAL '
                 ');')
  cursor.execute('CREATE INDEX objects_on_imagefile ON objects(imagefile);')
  cursor.execute('CREATE INDEX objects_on_objectid ON objects(objectid);')


def createTableProperties (cursor):
  cursor.execute('CREATE TABLE properties '
                '(id INTEGER PRIMARY KEY, '
                 'objectid INTEGER, '
                 'key TEXT, '
                 'value TEXT '
                 ');')
  cursor.execute('CREATE INDEX properties_on_key ON properties(key);')
  cursor.execute('CREATE INDEX properties_on_key_and_value ON properties(key,value);')
  cursor.execute('CREATE INDEX properties_on_objectid ON properties(objectid);')


def createTablePolygons (cursor):
  cursor.execute('CREATE TABLE polygons '
                '(id INTEGER PRIMARY KEY, '
                 'objectid INTEGER, '
                 'x INTEGER, '
                 'y INTEGER, '
                 'name TEXT '
                 ');')
  cursor.execute('CREATE INDEX polygons_on_id ON polygons(id);')
  cursor.execute('CREATE INDEX polygons_on_objectid ON polygons(objectid);')


def createTableMatches (cursor):
  cursor.execute('CREATE TABLE matches '
                '(id INTEGER PRIMARY KEY, '
                 'match INTEGER, '
                 'objectid INTEGER '
                 ');')
  cursor.execute('CREATE INDEX matches_on_match ON matches(match);')


def createDb (conn):
  ''' Creates all the necessary tables and indexes. '''

  cursor = conn.cursor()
  conn.execute('PRAGMA user_version = 4')  # This is version 4.
  createTableImages(cursor)
  createTableObjects(cursor)
  createTableProperties(cursor)
  createTablePolygons(cursor)
  createTableMatches(cursor)


def makeTimeString (time):
  ''' Write a time string in Shuffler format.
  Args:      time -- a datetime.datetime object.
  Returns:   string.
  '''
  return datetime.strftime(time, '%Y-%m-%d %H:%M:%S.%f')


def parseTimeString (timestring):
  ''' Parses the Shuffler format.
  Args:      timestring -- a string object
  Returns:   datetime.datetime object
  '''
  return datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S.%f')


# def parseIdatafaTimeString (timestring):
#   # E.g. 2016/04/25 18:00:25
#   return datetime.strptime(timestring, '%Y/%m/%d %H:%M:%S')


def objectField (entry, field):
  ''' Convenience function to access by field name. '''

  if field == 'objectid':  return entry[0]
  if field == 'imagefile': return entry[1] 
  if field == 'x1':        return entry[2]
  if field == 'y1':        return entry[3]
  if field == 'width':     return entry[4]
  if field == 'height':    return entry[5]
  if field == 'name':      return entry[6] 
  if field == 'score':     return entry[7] 
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
      return [bbox[1], bbox[0], bbox[3]+bbox[1]-1, bbox[2]+bbox[0]-1]
  raise KeyError('No field "%s" in object entry %s' % (field, entry))

def imageField (entry, field):
  ''' Convenience function to access by field name. '''

  if field == 'imagefile': return entry[0]
  if field == 'width':     return entry[1]
  if field == 'height':    return entry[2]
  if field == 'maskfile':  return entry[3]
  if field == 'timestamp': return entry[4]
  if field == 'name':      return entry[5]
  if field == 'score':     return entry[6]
  raise KeyError('No field "%s" in image entry %s' % (field, entry))

def polygonField (entry, field):
  ''' Convenience function to access by field name. '''

  if field == 'id':        return entry[0]
  if field == 'objectid':  return entry[1]
  if field == 'x':         return entry[2]
  if field == 'y':         return entry[3]
  if field == 'name':      return entry[4]
  raise KeyError('No field "%s" in polygon entry %s' % (field, entry))


def deleteObject (cursor, objectid):
  ''' Delete entries from all tables associated with the object.
  If the object does not exist, raises KeyError.
  '''
  cursor.execute('SELECT COUNT(1) FROM objects WHERE objectid=?;', (objectid,))
  if cursor.fetchone()[0] == 0:
    raise KeyError('Can not delete objectid %d, as it is not in the database' % objectid)
  cursor.execute('DELETE FROM objects WHERE objectid=?;', (objectid,))
  cursor.execute('DELETE FROM matches WHERE objectid=?;', (objectid,))
  cursor.execute('DELETE FROM polygons WHERE objectid=?;', (objectid,))
  cursor.execute('DELETE FROM properties WHERE objectid=?;', (objectid,))


def deleteImage (cursor, imagefile):
  ''' Delete entries from all tables associated with the imagefile and all objects in this imagefile.
  If the image does not exist, raises KeyError.
  '''
  cursor.execute('SELECT COUNT(1) FROM images WHERE imagefile=?;', (imagefile,))
  if cursor.fetchone()[0] == 0:
    raise KeyError('Can not delete imagefile %s, as it is not in the database' % imagefile)
  cursor.execute('DELETE FROM matches WHERE objectid IN (SELECT objectid FROM objects WHERE imagefile=?);', (imagefile,))
  cursor.execute('DELETE FROM polygons WHERE objectid IN (SELECT objectid FROM objects WHERE imagefile=?);', (imagefile,))
  cursor.execute('DELETE FROM properties WHERE objectid IN (SELECT objectid FROM objects WHERE imagefile=?);', (imagefile,))
  cursor.execute('DELETE FROM objects WHERE imagefile=?;', (imagefile,))
  cursor.execute('DELETE FROM images WHERE imagefile=?;', (imagefile,))

