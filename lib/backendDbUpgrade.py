import os, os.path as op
import logging
import sqlite3
import datetime
import argparse
import progressbar
from pprint import pformat

import backendDb
from util import copyWithBackup


def upgradeV3toV4(cursor):
  ''' Upgrade the schema. '''

  # Drop all indexes.
  cursor.execute('SELECT name FROM sqlite_master WHERE type == "index"')
  for index_name, in cursor.fetchall():
    try:
      logging.debug('Dropping index: %s' % index_name)
      cursor.execute('DROP INDEX %s' % index_name)
    except sqlite3.OperationalError:
      logging.debug('Can not drop this index.')
  # Images.
  cursor.execute('ALTER TABLE images RENAME TO images_old')
  backendDb.createTableImages(cursor)
  cursor.execute('INSERT INTO images(imagefile, width, height, maskfile, timestamp) '
                 'SELECT imagefile, width, height, maskfile, time FROM images_old;')
  cursor.execute('DROP TABLE images_old;')
  # Properties.
  backendDb.createTableProperties(cursor)
  cursor.execute("INSERT INTO properties(objectid, key, value) "
                 "SELECT id, 'yaw', yaw     FROM cars WHERE yaw   IS NOT NULL;")
  cursor.execute("INSERT INTO properties(objectid, key, value) "
                 "SELECT id, 'pitch', pitch FROM cars WHERE pitch IS NOT NULL;")
  cursor.execute("INSERT INTO properties(objectid, key, value) "
                 "SELECT id, 'color', color FROM cars WHERE color IS NOT NULL;")
  # Objects.
  backendDb.createTableObjects(cursor)
  cursor.execute('INSERT INTO objects(objectid,imagefile,x1,y1,width,height,name,score) '
                 'SELECT id,imagefile,x1,y1,width,height,name,score FROM cars;')
  cursor.execute('DROP TABLE cars;')
  # Matches.
  cursor.execute('ALTER TABLE matches RENAME TO matches_old')
  backendDb.createTableMatches(cursor)
  cursor.execute('INSERT INTO matches(id,match,objectid) '
                 'SELECT id,match,carid FROM matches_old;')
  cursor.execute('DROP TABLE matches_old;')
  # Polygons.
  cursor.execute('ALTER TABLE polygons RENAME TO polygons_old')
  backendDb.createTablePolygons(cursor)
  cursor.execute('INSERT INTO polygons(id,objectid,x,y) '
                 'SELECT id,carid,x,y FROM polygons_old;')
  cursor.execute('DROP TABLE polygons_old;')
  # Info on indexes.
  cursor.execute('SELECT name FROM sqlite_master WHERE type == "index"')
  logging.debug('All indexes: %s' % pformat(cursor.fetchall()))


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Upgrade schema from v3 to v4.')
  parser.add_argument('--in_db_path', required=True)
  parser.add_argument('--out_db_path')
  parser.add_argument('--no_backup', action='store_true', help='Do not backup')
  parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
    help='Log debug (10), info (20), warning (30), error (40).')
  args = parser.parse_args()

  progressbar.streams.wrap_stderr()
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  # Maybe back up.
  if args.out_db_path is not None:
    copyWithBackup(args.in_db_path, args.out_db_path)
    db_path = args.out_db_path
  else:
    db_path = args.in_db_path

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  upgradeV3toV4(cursor)
  
  conn.commit()
  conn.close()
