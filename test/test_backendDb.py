import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import random
import logging
import sqlite3
import shutil
import unittest

from lib import backendDb


class TestEmptyDb (unittest.TestCase):

  def setUp (self):
    self.conn = sqlite3.connect(':memory:')  # in RAM
    backendDb.createDb(self.conn)

  def tearDown (self):
    self.conn.close()

  def _test_table(self, cursor, table, cols_gt):

    # Test table exists.
    cursor.execute('SELECT count(*) FROM sqlite_master WHERE name=? AND type="table"', (table,))
    assert cursor.fetchone()[0] == 1
    
    # Test cols.
    cursor.execute('PRAGMA table_info(%s)' % table)
    cols_actual = [x[1] for x in cursor.fetchall()]
    self.assertEqual (set(cols_actual), set(cols_gt))

  def test_schema (self):
    cursor = self.conn.cursor()

    self._test_table(cursor, 'images',
      ['imagefile', 'maskfile', 'width', 'height', 'timestamp', 'score', 'name'])
    self._test_table(cursor, 'objects',
      ['objectid', 'imagefile', 'x1', 'y1', 'width', 'height', 'score', 'name'])
    self._test_table(cursor, 'matches', ['id', 'objectid', 'match'])
    self._test_table(cursor, 'properties', ['id', 'objectid', 'key', 'value'])
    self._test_table(cursor, 'polygons', ['id', 'objectid', 'x', 'y', 'name'])



class TestGetFields (unittest.TestCase):

  def setUp (self):
    self.conn = sqlite3.connect('file:databases/micro1_v4.db?mode=ro', uri=True)

  def tearDown (self):
    self.conn.close()

  def test_objectFields(self):
    cursor = self.conn.cursor()
    cursor.execute('SELECT * FROM objects WHERE objectid=1')
    entry = cursor.fetchone()

    self.assertEqual(backendDb.objectField(entry, 'objectid'), 1, str(entry))
    self.assertEqual(backendDb.objectField(entry, 'imagefile'), 'images/000000.jpg', str(entry))
    self.assertEqual(backendDb.objectField(entry, 'x1'), 225, str(entry))
    self.assertEqual(backendDb.objectField(entry, 'y1'), 134, str(entry))
    self.assertEqual(backendDb.objectField(entry, 'width'), 356, str(entry))
    self.assertEqual(backendDb.objectField(entry, 'height'), 377, str(entry))
    self.assertEqual(backendDb.objectField(entry, 'name'), 'car', str(entry))
    self.assertAlmostEqual(backendDb.objectField(entry, 'score'), 0.606193, msg=str(entry))
    with self.assertRaises(KeyError):
      backendDb.objectField(entry, 'dummy')

  def test_imageFields(self):
    cursor = self.conn.cursor()
    cursor.execute('SELECT * FROM images WHERE imagefile="images/000000.jpg"')
    entry = cursor.fetchone()

    self.assertEqual(backendDb.imageField(entry, 'imagefile'), 'images/000000.jpg', str(entry))
    self.assertEqual(backendDb.imageField(entry, 'width'), 800, str(entry))
    self.assertEqual(backendDb.imageField(entry, 'height'), 700, str(entry))
    self.assertEqual(backendDb.imageField(entry, 'maskfile'), 'masks/000000.png', str(entry))
    self.assertEqual(backendDb.imageField(entry, 'timestamp'), '2018-09-24 12:22:48.534685', str(entry))
    with self.assertRaises(KeyError):
      backendDb.imageField(entry, 'dummy')

  def test_polygonFields(self):
    cursor = self.conn.cursor()
    cursor.execute('SELECT * FROM polygons WHERE id=1')
    entry = cursor.fetchone()

    self.assertEqual(backendDb.polygonField(entry, 'id'), 1, str(entry))
    self.assertEqual(backendDb.polygonField(entry, 'objectid'), 2, str(entry))
    self.assertEqual(backendDb.polygonField(entry, 'x'), 97, str(entry))
    self.assertEqual(backendDb.polygonField(entry, 'y'), 296, str(entry))
    self.assertEqual(backendDb.polygonField(entry, 'name'), None, str(entry))
    with self.assertRaises(KeyError):
      backendDb.polygonField(entry, 'dummy')


class TestDeleteObject (unittest.TestCase):

  def setUp (self):
    shutil.copyfile('databases/micro1_v4.db', 'databases/micro1_v4.temp.db')
    self.conn = sqlite3.connect('databases/micro1_v4.temp.db')
    self.cursor = self.conn.cursor()

  def tearDown (self):
    self.conn.close()
    os.remove('databases/micro1_v4.temp.db')

  def test_deleteObject0(self):
    with self.assertRaises(KeyError):
      backendDb.deleteObject (self.cursor, objectid=0)

  def test_deleteObject1(self):
    backendDb.deleteObject (self.cursor, objectid=1)

    self.cursor.execute('SELECT objectid FROM objects')
    objectids = self.cursor.fetchall()
    self.assertEqual(objectids, [(2,), (3,)], str(objectids))

    self.cursor.execute('SELECT DISTINCT(objectid) FROM properties')
    self.assertEqual(self.cursor.fetchall(), [(2,)])

    self.cursor.execute('SELECT DISTINCT(objectid) FROM matches')
    self.assertEqual(self.cursor.fetchall(), [(2,)])

    self.cursor.execute('SELECT DISTINCT(objectid) FROM polygons')
    self.assertEqual(self.cursor.fetchall(), [(2,)])

  def test_deleteObject2(self):
    backendDb.deleteObject (self.cursor, objectid=2)

    self.cursor.execute('SELECT objectid FROM objects')
    objectids = self.cursor.fetchall()
    self.assertEqual(objectids, [(1,), (3,)], str(objectids))

    self.cursor.execute('SELECT DISTINCT(objectid) FROM properties')
    self.assertEqual(self.cursor.fetchall(), [(1,)])

    self.cursor.execute('SELECT DISTINCT(objectid) FROM matches')
    self.assertEqual(self.cursor.fetchall(), [(1,)])

    self.cursor.execute('SELECT DISTINCT(objectid) FROM polygons')
    self.assertEqual(self.cursor.fetchall(), [])




if __name__ == '__main__':
  logging.basicConfig (level=logging.ERROR)
  unittest.main()
