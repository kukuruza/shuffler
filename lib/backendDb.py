import os, os.path as op
import logging
import sqlite3
import datetime


def doesTableExist (cursor, table):
    cursor.execute('''SELECT count(*) FROM sqlite_master 
                      WHERE name=? AND type='table';''', (table,))
    return cursor.fetchone()[0] != 0

def isColumnInTable (cursor, table, column):
    if not doesTableExist(cursor, table):
        raise Exception ('table %s does not exist' % table)
    cursor.execute('PRAGMA table_info(%s)' % table)
    return column in [x[1] for x in cursor.fetchall()]

def createTableImages (cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS images
                     (imagefile TEXT PRIMARY KEY, 
                      src TEXT, 
                      width INTEGER, 
                      height INTEGER,
                      maskfile TEXT,
                      time TIMESTAMP
                      );''')
    cursor.execute('CREATE INDEX images_on_imagefile ON images(imagefile);')

def createTableCars (cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS cars
                     (id INTEGER PRIMARY KEY,
                      imagefile TEXT, 
                      name TEXT, 
                      x1 INTEGER,
                      y1 INTEGER,
                      width INTEGER, 
                      height INTEGER,
                      score REAL,
                      yaw REAL,
                      pitch REAL,
                      color TEXT
                      );''')
    cursor.execute('CREATE INDEX cars_on_imagefile ON cars(imagefile);')
    cursor.execute('CREATE INDEX cars_on_carid ON cars(id);')

def createTablePolygons (cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS polygons
                     (id INTEGER PRIMARY KEY,
                      carid INTEGER, 
                      x INTEGER,
                      y INTEGER
                      );''')
    cursor.execute('CREATE INDEX polygons_on_id ON polygons(id);')
    cursor.execute('CREATE INDEX polygons_on_carid ON polygons(carid);')

def createTableMatches (cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS matches
                     (id INTEGER PRIMARY KEY,
                      match INTEGER,
                      carid INTEGER
                     );''')
    cursor.execute('CREATE INDEX matches_on_match ON matches(match);')

def create(conn):
    cursor = conn.cursor()
    conn.execute('PRAGMA user_version = 3')
    createTablePolygons(cursor)
    createTableImages(cursor)
    createTableCars(cursor)
    createTableMatches(cursor)

def makeTimeString (time):
    ''' Write a string in my format.
    Args: time -- datetime object
    '''
    return datetime.strftime(time, '%Y-%m-%d %H:%M:%S.%f')

def parseTimeString (timestring):
    return datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S.%f')

def parseIdatafaTimeString (timestring):
    # E.g. 2016/04/25 18:00:25
    return datetime.strptime(timestring, '%Y/%m/%d %H:%M:%S')

def loadToMemory(in_db_path):
    if not op.exists(in_db_path):
      raise Exception('Input database provided but does not exist: %s' % in_db_path)
    conn_in = sqlite3.connect(in_db_path)
    conn = sqlite3.connect(':memory:') # create a memory database
    query = ''.join(line for line in conn_in.iterdump())
    # Dump input database in the one in memory. 
    conn.executescript(query)
    return conn


def carField (car, field):
    ''' all knowledge about 'cars' table is here '''
    if field == 'id':        return car[0]
    if field == 'imagefile': return car[1] 
    if field == 'name':      return car[2] 
    if field == 'x1':        return car[3]
    if field == 'y1':        return car[4]
    if field == 'width':     return car[5]
    if field == 'height':    return car[6]
    if field == 'score':     return car[7]
    if field == 'yaw':       return car[8]
    if field == 'pitch':     return car[9]
    if field == 'color':     return car[10]

    if field == 'bbox':      
        return list(car[3:7])
    if field == 'roi':
        bbox = list(car[3:7])
        return [bbox[1], bbox[0], bbox[3]+bbox[1]-1, bbox[2]+bbox[0]-1]
    return None


def imageField (image, field):
    if field == 'imagefile': return image[0] 
    if field == 'src':       return image[1] 
    if field == 'width':     return image[2] 
    if field == 'height':    return image[3] 
    if field == 'maskfile':  return image[4] 
    if field == 'time':      return image[5] 
    return None


def polygonField (polygon, field):
    if field == 'id':        return polygon[0]
    if field == 'carid':     return polygon[1]
    if field == 'x':         return polygon[2]
    if field == 'y':         return polygon[3]
    return None


def deleteCar (cursor, carid, has_polygons=False, has_matches=False):#, remove_matched=True):
  ''' Delete all information about a car.
  If the car does not exist, operation has no effect.
  Args:
    remove_matches:  if True, also delete all cars matched to the current car.
  '''
  cursor.execute('DELETE FROM cars WHERE id=?;', (carid,))
  if has_matches:
    cursor.execute('''SELECT carid FROM matches WHERE carid IN
                      (SELECT match FROM matches WHERE carid=?)''', (carid,))
    carids_matched = cursor.fetchall()
    logging.debug ('Car is in %d matches.' % len(carids_matched))
    cursor.execute('DELETE FROM matches WHERE carid=?;', (carid,))
    #if remove_matched:
    #  for carid_matched, in carids_matched:
    #    if carid_matched != carid:
    #      deleteCar(cursor, carid_matched, has_polygons, has_matches, remove_matched=False)
  if has_polygons:
    cursor.execute('DELETE FROM polygons WHERE carid=?;', (carid,))


def deleteCars (cursor, carids):
  has_polygons = doesTableExist(cursor, 'polygons')
  has_matches = doesTableExist(cursor, 'matches')
  for carid, in ProgressBar()(carids):
    deleteCar (cursor, carid, has_polygons=has_polygons, has_matches=has_matches)


