import os
import sqlite3
import shutil
import unittest
import tempfile
import numpy as np
import progressbar
import nose

from lib.backend import backendDb
from lib.utils import testUtils


class TestCars(unittest.TestCase):

    CARS_DB_PATH = 'testdata/cars/micro1_v5.db'

    def setUp(self):
        self.temp_db_path = tempfile.NamedTemporaryFile().name
        shutil.copyfile(TestCars.CARS_DB_PATH, self.temp_db_path)

        self.conn = sqlite3.connect(self.temp_db_path)
        self.cursor = self.conn.cursor()

    def tearDown(self):
        self.conn.close()
        os.remove(self.temp_db_path)

    def test_objectFields(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM objects WHERE objectid=1')
        entry = cursor.fetchone()

        self.assertEqual(backendDb.objectField(entry, 'objectid'), 1,
                         str(entry))
        self.assertEqual(backendDb.objectField(entry, 'imagefile'),
                         'images/000000.jpg', str(entry))
        self.assertEqual(backendDb.objectField(entry, 'x1'), 225, str(entry))
        self.assertEqual(backendDb.objectField(entry, 'y1'), 134, str(entry))
        self.assertEqual(backendDb.objectField(entry, 'width'), 356,
                         str(entry))
        self.assertEqual(backendDb.objectField(entry, 'height'), 377,
                         str(entry))
        self.assertEqual(backendDb.objectField(entry, 'name'), 'car',
                         str(entry))
        self.assertAlmostEqual(backendDb.objectField(entry, 'score'),
                               0.606193,
                               msg=str(entry))
        with self.assertRaises(KeyError):
            backendDb.objectField(entry, 'dummy')

        # Multiple fields.
        self.assertEqual(backendDb.objectFields(entry, ['objectid', 'x1']),
                         [1, 225], str(entry))
        with self.assertRaises(KeyError):
            self.assertEqual(backendDb.objectFields(entry, ['x1', 'dummy']))

    def test_imageFields(self):
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM images WHERE imagefile="images/000000.jpg"')
        entry = cursor.fetchone()

        self.assertEqual(backendDb.imageField(entry, 'imagefile'),
                         'images/000000.jpg', str(entry))
        self.assertEqual(backendDb.imageField(entry, 'width'), 800, str(entry))
        self.assertEqual(backendDb.imageField(entry, 'height'), 700,
                         str(entry))
        self.assertEqual(backendDb.imageField(entry, 'maskfile'),
                         'masks/000000.png', str(entry))
        self.assertEqual(backendDb.imageField(entry, 'timestamp'),
                         '2018-09-24 12:22:48.534685', str(entry))
        self.assertEqual(backendDb.imageField(entry, 'name'), None, str(entry))
        self.assertEqual(backendDb.imageField(entry, 'score'), None,
                         str(entry))
        with self.assertRaises(KeyError):
            backendDb.imageField(entry, 'dummy')

        # Multiple fields.
        self.assertEqual(backendDb.imageFields(entry, ['name', 'width']),
                         [None, 800], str(entry))
        with self.assertRaises(KeyError):
            self.assertEqual(backendDb.imageFields(entry, ['name', 'dummy']))

    def test_polygonFields(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM polygons WHERE id=1')
        entry = cursor.fetchone()

        self.assertEqual(backendDb.polygonField(entry, 'id'), 1, str(entry))
        self.assertEqual(backendDb.polygonField(entry, 'objectid'), 2,
                         str(entry))
        self.assertEqual(backendDb.polygonField(entry, 'x'), 97, str(entry))
        self.assertEqual(backendDb.polygonField(entry, 'y'), 296, str(entry))
        self.assertEqual(backendDb.polygonField(entry, 'name'), None,
                         str(entry))
        with self.assertRaises(KeyError):
            backendDb.polygonField(entry, 'dummy')

        # Multiple fields.
        self.assertEqual(backendDb.polygonFields(entry, ['id', 'x']), [1, 97],
                         str(entry))
        with self.assertRaises(KeyError):
            self.assertEqual(backendDb.polygonFields(entry, ['id', 'dummy']))

    def test_getColumnsInTable(self):
        result = backendDb.getColumnsInTable(self.cursor, 'objects')
        self.assertEqual(result, [
            'objectid', 'imagefile', 'x1', 'y1', 'width', 'height', 'name',
            'score'
        ])

    def test_delete_imagefile_nonexistent(self):
        with self.assertRaises(KeyError):
            backendDb.deleteImage(self.cursor, imagefile='not_existent')

    def test_delete_imagefile000000(self):
        backendDb.deleteImage(self.cursor, imagefile='images/000000.jpg')

        self.cursor.execute('SELECT imagefile FROM images')
        imagefiles = self.cursor.fetchall()
        self.assertEqual(imagefiles, [('images/000001.jpg', ),
                                      ('images/000002.jpg', )],
                         str(imagefiles))

        self.cursor.execute('SELECT objectid FROM objects')
        objectids = self.cursor.fetchall()
        self.assertEqual(objectids, [(2, ), (3, )], str(objectids))

        self.cursor.execute('SELECT DISTINCT(objectid) FROM properties')
        self.assertEqual(self.cursor.fetchall(), [(2, ), (3, )])

        self.cursor.execute('SELECT DISTINCT(objectid) FROM matches')
        self.assertEqual(self.cursor.fetchall(), [(2, )])

        self.cursor.execute('SELECT DISTINCT(objectid) FROM polygons')
        self.assertEqual(self.cursor.fetchall(), [(2, )])

    def test_delete_imagefile000001(self):
        backendDb.deleteImage(self.cursor, imagefile='images/000001.jpg')

        self.cursor.execute('SELECT imagefile FROM images')
        imagefiles = self.cursor.fetchall()
        self.assertEqual(imagefiles, [('images/000000.jpg', ),
                                      ('images/000002.jpg', )],
                         str(imagefiles))

        self.cursor.execute('SELECT objectid FROM objects')
        objectids = self.cursor.fetchall()
        self.assertEqual(objectids, [(1, )], str(objectids))

        self.cursor.execute('SELECT DISTINCT(objectid) FROM properties')
        self.assertEqual(self.cursor.fetchall(), [(1, )])

        self.cursor.execute('SELECT DISTINCT(objectid) FROM matches')
        self.assertEqual(self.cursor.fetchall(), [(1, )])

        self.cursor.execute('SELECT DISTINCT(objectid) FROM polygons')
        self.assertEqual(self.cursor.fetchall(), [])

    def test_deleteObject0(self):
        with self.assertRaises(KeyError):
            backendDb.deleteObject(self.cursor, objectid=0)

    def test_deleteObject1(self):
        backendDb.deleteObject(self.cursor, objectid=1)

        self.cursor.execute('SELECT objectid FROM objects')
        objectids = self.cursor.fetchall()
        self.assertEqual(objectids, [(2, ), (3, )], str(objectids))

        self.cursor.execute('SELECT DISTINCT(objectid) FROM properties')
        self.assertEqual(self.cursor.fetchall(), [(2, ), (3, )])

        self.cursor.execute('SELECT DISTINCT(objectid) FROM matches')
        self.assertEqual(self.cursor.fetchall(), [(2, )])

        self.cursor.execute('SELECT DISTINCT(objectid) FROM polygons')
        self.assertEqual(self.cursor.fetchall(), [(2, )])

    def test_deleteObject2(self):
        backendDb.deleteObject(self.cursor, objectid=2)

        self.cursor.execute('SELECT objectid FROM objects')
        objectids = self.cursor.fetchall()
        self.assertEqual(objectids, [(1, ), (3, )], str(objectids))

        self.cursor.execute('SELECT DISTINCT(objectid) FROM properties')
        self.assertEqual(self.cursor.fetchall(), [(1, ), (3, )])

        self.cursor.execute('SELECT DISTINCT(objectid) FROM matches')
        self.assertEqual(self.cursor.fetchall(), [(1, )])

        self.cursor.execute('SELECT DISTINCT(objectid) FROM polygons')
        self.assertEqual(self.cursor.fetchall(), [])


class Test_updateObjectTransform_emptyDb(testUtils.Test_emptyDb):
    def test_noPreviousTransform(self):
        c = self.conn.cursor()

        transform = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        backendDb.updateObjectTransform(c, 0, transform)

        c.execute('SELECT value FROM properties WHERE key="ky"')
        ky = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="kx"')
        kx = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="by"')
        by = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="bx"')
        bx = c.fetchall()
        self.assertEqual((len(ky), len(kx), len(by), len(bx)), (1, 1, 1, 1))
        self.assertEqual(float(ky[0][0]), 1.)
        self.assertEqual(float(kx[0][0]), 1.)
        self.assertEqual(float(by[0][0]), 0.)
        self.assertEqual(float(bx[0][0]), 0.)

    def test_withPreviousTransform(self):
        c = self.conn.cursor()

        transform0 = np.array([[2., 0., 0.], [0., 1., 1.], [0., 0., 1.]])
        c.execute(
            'INSERT INTO properties(objectid,key,value) '
            'VALUES(0,"ky",?), (0,"kx",?), (0,"by",?), (0,"bx",?)',
            (transform0[0, 0], transform0[1, 1], transform0[0, 2],
             transform0[1, 2]))

        transform1 = np.array([[1., 0., 1.], [0., 2., 1.], [0., 0., 1.]])
        backendDb.updateObjectTransform(c, 0, transform1)

        c.execute('SELECT value FROM properties WHERE key="ky"')
        ky = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="kx"')
        kx = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="by"')
        by = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="bx"')
        bx = c.fetchall()
        self.assertEqual((len(ky), len(kx), len(by), len(bx)), (1, 1, 1, 1))
        transform2 = np.array([[float(ky[0][0]), 0.,
                                float(by[0][0])],
                               [0., float(kx[0][0]),
                                float(bx[0][0])], [0., 0., 1.]])

        np.testing.assert_array_equal(np.matmul(transform1, transform0),
                                      transform2)


class TestUpgradeV4toV5Cars(testUtils.Test_emptyDb):

    CARS_DB_V4_PATH = 'testdata/cars/micro1_v5.db'
    CARS_DB_V5_PATH = 'testdata/cars/micro1_v5.db'

    def setUp(self):
        self.temp_db_v4_path = tempfile.NamedTemporaryFile().name
        self.temp_db_v5_path = tempfile.NamedTemporaryFile().name

        shutil.copyfile(self.CARS_DB_V4_PATH, self.temp_db_v4_path)
        shutil.copyfile(self.CARS_DB_V5_PATH, self.temp_db_v5_path)

        self.conn = sqlite3.connect(self.temp_db_v4_path)
        backendDb.upgradeV4toV5(self.conn.cursor())

    def tearDown(self):
        self.conn.close()
        os.remove(self.temp_db_v4_path)
        os.remove(self.temp_db_v5_path)

    def _test_table_contents(self, cursor, table):
        ''' Compare the contents of "table" between "main" and "gt" schema names. '''

        cursor.execute('SELECT * FROM %s' % table)
        entries_main = cursor.fetchall()
        cursor.execute('ATTACH "%s" AS gt' % self.temp_db_v5_path)
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

    def test_indexNames(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type == "index"')
        entries_main = cursor.fetchall()
        cursor.execute('ATTACH "%s" AS gt' % self.temp_db_v5_path)
        cursor.execute('SELECT name FROM sqlite_master WHERE type == "index"')
        entries_gt = cursor.fetchall()
        self.assertEqual(entries_main, entries_gt)

        # TODO: Test that indexes actually have the same info.


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
