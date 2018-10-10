import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import random
import logging
import sqlite3
import unittest
import shutil

from lib import backendDbUpgrade
from test_backendDb import TestEmptyDb


class TestUpgradeV3toV4 (TestEmptyDb):

  def setUp (self):
    shutil.copyfile('databases/micro1_v3.db', 'databases/micro1_v3.temp.db')
    shutil.copyfile('databases/micro1_v4.db', 'databases/micro1_v4.temp.db')
    self.conn = sqlite3.connect('databases/micro1_v3.temp.db')
    backendDbUpgrade.upgradeV3toV4(self.conn.cursor())

  def tearDown (self):
    self.conn.close()
    os.remove('databases/micro1_v3.temp.db')
    os.remove('databases/micro1_v4.temp.db')

  def _test_table_contents(self, cursor, table):
    ''' Compare the contents of "table" between "main" and "gt" schema names. '''

    cursor.execute('SELECT * FROM main.%s' % table)
    entries_main = cursor.fetchall()
    cursor.execute('ATTACH "databases/micro1_v4.temp.db" AS gt')
    cursor.execute('SELECT * FROM gt.%s' % table)
    entries_gt = cursor.fetchall()
    self.assertEqual (entries_main, entries_gt)

  def test_compareContentImages (self):
    self._test_table_contents(self.conn.cursor(), 'images')

  def test_compareContentObjects (self):
    self._test_table_contents(self.conn.cursor(), 'objects')

  def test_compareContentMatches (self):
    self._test_table_contents(self.conn.cursor(), 'matches')

  def test_compareContentPolygons (self):
    self._test_table_contents(self.conn.cursor(), 'polygons')

  def test_compareContentProperties (self):
    self._test_table_contents(self.conn.cursor(), 'properties')


if __name__ == '__main__':
    logging.basicConfig (level=logging.INFO)
    unittest.main()
