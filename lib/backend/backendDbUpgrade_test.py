import os, sys
import random
import logging
import sqlite3
import unittest
import shutil
import tempfile
import nose

from lib.backend import backendDbUpgrade
from lib.utils import testUtils

CARS_DB_V3_PATH = 'testdata/cars/micro1_v3.db'
CARS_DB_V4_PATH = 'testdata/cars/micro1_v4.db'


class TestUpgradeV3toV4Cars(testUtils.Test_emptyDb):
    def setUp(self):
        self.temp_db_v3_path = tempfile.NamedTemporaryFile().name
        self.temp_db_v4_path = tempfile.NamedTemporaryFile().name

        shutil.copyfile(CARS_DB_V3_PATH, self.temp_db_v3_path)
        shutil.copyfile(CARS_DB_V4_PATH, self.temp_db_v4_path)

        self.conn = sqlite3.connect(self.temp_db_v3_path)
        backendDbUpgrade.upgradeV3toV4(self.conn.cursor())

    def tearDown(self):
        self.conn.close()
        os.remove(self.temp_db_v3_path)
        os.remove(self.temp_db_v4_path)

    def _test_table_contents(self, cursor, table):
        ''' Compare the contents of "table" between "main" and "gt" schema names. '''

        cursor.execute('SELECT * FROM main.%s' % table)
        entries_main = cursor.fetchall()
        cursor.execute('ATTACH "%s" AS gt' % self.temp_db_v4_path)
        cursor.execute('SELECT * FROM gt.%s' % table)
        entries_gt = cursor.fetchall()
        self.assertEqual(entries_main, entries_gt)

    def test_compareContentImages(self):
        self._test_table_contents(self.conn.cursor(), 'images')

    def test_compareContentObjects(self):
        self._test_table_contents(self.conn.cursor(), 'objects')

    def test_compareContentMatches(self):
        self._test_table_contents(self.conn.cursor(), 'matches')

    def test_compareContentPolygons(self):
        self._test_table_contents(self.conn.cursor(), 'polygons')

    def test_compareContentProperties(self):
        self._test_table_contents(self.conn.cursor(), 'properties')


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
