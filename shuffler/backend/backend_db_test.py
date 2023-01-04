import os
import sqlite3
import shutil
import tempfile
import numpy as np
import progressbar
import nose

from shuffler.backend import backend_db
from shuffler.utils import testing as testing_utils


class TestFieldGetters_Cars(testing_utils.Test_carsDb):

    def test_objectField(self):
        c = self.conn.cursor()
        c.execute('SELECT * FROM objects WHERE objectid=1')
        entry = c.fetchone()

        self.assertEqual(backend_db.objectField(entry, 'objectid'),
                         1,
                         msg=str(entry))
        self.assertEqual(backend_db.objectField(entry, 'imagefile'),
                         'images/000000.jpg',
                         msg=str(entry))
        self.assertEqual(backend_db.objectField(entry, 'x1'),
                         225.1,
                         msg=str(entry))
        self.assertEqual(backend_db.objectField(entry, 'y1'),
                         134.2,
                         msg=str(entry))
        self.assertEqual(backend_db.objectField(entry, 'width'),
                         356.3,
                         msg=str(entry))
        self.assertEqual(backend_db.objectField(entry, 'height'),
                         377.4,
                         msg=str(entry))
        self.assertEqual(backend_db.objectField(entry, 'name'),
                         'car',
                         msg=str(entry))
        self.assertAlmostEqual(backend_db.objectField(entry, 'score'),
                               0.606193,
                               msg=str(entry))
        with self.assertRaises(KeyError):
            backend_db.objectField(entry, 'dummy')

        # Multiple fields.
        self.assertEqual(backend_db.objectFields(entry, ['objectid', 'x1']),
                         [1, 225.1], str(entry))
        with self.assertRaises(KeyError):
            self.assertEqual(backend_db.objectFields(entry, ['x1', 'dummy']))

    def test_imageFields(self):
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM images WHERE imagefile="images/000000.jpg"')
        entry = cursor.fetchone()

        self.assertEqual(backend_db.imageField(entry, 'imagefile'),
                         'images/000000.jpg',
                         msg=str(entry))
        self.assertEqual(backend_db.imageField(entry, 'width'),
                         800,
                         msg=str(entry))
        self.assertEqual(backend_db.imageField(entry, 'height'),
                         700,
                         msg=str(entry))
        self.assertEqual(backend_db.imageField(entry, 'maskfile'),
                         'masks/000000.png',
                         msg=str(entry))
        self.assertEqual(backend_db.imageField(entry, 'timestamp'),
                         '2018-09-24 12:22:48.534685',
                         msg=str(entry))
        self.assertEqual(backend_db.imageField(entry, 'name'),
                         None,
                         msg=str(entry))
        self.assertEqual(backend_db.imageField(entry, 'score'),
                         None,
                         msg=str(entry))
        with self.assertRaises(KeyError):
            backend_db.imageField(entry, 'dummy')

        # Multiple fields.
        self.assertEqual(backend_db.imageFields(entry, ['name', 'width']),
                         [None, 800], str(entry))
        with self.assertRaises(KeyError):
            self.assertEqual(backend_db.imageFields(entry, ['name', 'dummy']))

    def test_polygonFields(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM polygons WHERE id=1')
        entry = cursor.fetchone()

        self.assertEqual(backend_db.polygonField(entry, 'id'),
                         1,
                         msg=str(entry))
        self.assertEqual(backend_db.polygonField(entry, 'objectid'),
                         2,
                         msg=str(entry))
        self.assertEqual(backend_db.polygonField(entry, 'x'),
                         97.1,
                         msg=str(entry))
        self.assertEqual(backend_db.polygonField(entry, 'y'),
                         296.0,
                         msg=str(entry))
        self.assertEqual(backend_db.polygonField(entry, 'name'),
                         None,
                         msg=str(entry))
        with self.assertRaises(KeyError):
            backend_db.polygonField(entry, 'dummy')

        # Multiple fields.
        self.assertEqual(backend_db.polygonFields(entry, ['id', 'x']),
                         [1, 97.1], str(entry))
        with self.assertRaises(KeyError):
            self.assertEqual(backend_db.polygonFields(entry, ['id', 'dummy']))

    def test_getColumnsInTable(self):
        c = self.conn.cursor()
        result = backend_db.getColumnsInTable(c, 'objects')
        self.assertEqual(result, [
            'objectid', 'imagefile', 'x1', 'y1', 'width', 'height', 'name',
            'score'
        ])


class TestDeleteImage_Cars(testing_utils.Test_carsDb):

    def test_delete_imagefile_nonexistent(self):
        c = self.conn.cursor()
        with self.assertRaises(KeyError):
            backend_db.deleteImage(c, imagefile='not_existent')

    def test_delete_imagefile000000(self):
        c = self.conn.cursor()
        backend_db.deleteImage(c, imagefile='images/000000.jpg')

        c.execute('SELECT imagefile FROM images')
        imagefiles = c.fetchall()
        self.assertEqual(imagefiles, [('images/000001.jpg', ),
                                      ('images/000002.jpg', )],
                         str(imagefiles))

        c.execute('SELECT objectid FROM objects')
        objectids = c.fetchall()
        self.assertEqual(objectids, [(2, ), (3, )], str(objectids))

        c.execute('SELECT DISTINCT(objectid) FROM properties')
        self.assertEqual(c.fetchall(), [(2, ), (3, )])

        c.execute('SELECT DISTINCT(objectid) FROM matches')
        self.assertEqual(c.fetchall(), [(2, )])

        c.execute('SELECT DISTINCT(objectid) FROM polygons')
        self.assertEqual(c.fetchall(), [(2, )])

    def test_delete_imagefile000001(self):
        c = self.conn.cursor()
        backend_db.deleteImage(c, imagefile='images/000001.jpg')

        c.execute('SELECT imagefile FROM images')
        imagefiles = c.fetchall()
        self.assertEqual(imagefiles, [('images/000000.jpg', ),
                                      ('images/000002.jpg', )],
                         str(imagefiles))

        c.execute('SELECT objectid FROM objects')
        objectids = c.fetchall()
        self.assertEqual(objectids, [(1, )], str(objectids))

        c.execute('SELECT DISTINCT(objectid) FROM properties')
        self.assertEqual(c.fetchall(), [(1, )])

        c.execute('SELECT DISTINCT(objectid) FROM matches')
        self.assertEqual(c.fetchall(), [(1, )])

        c.execute('SELECT DISTINCT(objectid) FROM polygons')
        self.assertEqual(c.fetchall(), [])


class TestDeleteObject_Cars(testing_utils.Test_carsDb):

    def test_deleteObject0(self):
        c = self.conn.cursor()
        with self.assertRaises(KeyError):
            backend_db.deleteObject(c, objectid=0)

    def test_deleteObject1(self):
        c = self.conn.cursor()
        backend_db.deleteObject(c, objectid=1)

        c.execute('SELECT objectid FROM objects')
        objectids = c.fetchall()
        self.assertEqual(objectids, [(2, ), (3, )], str(objectids))

        c.execute('SELECT DISTINCT(objectid) FROM properties')
        self.assertEqual(c.fetchall(), [(2, ), (3, )])

        c.execute('SELECT DISTINCT(objectid) FROM matches')
        self.assertEqual(c.fetchall(), [(2, )])

        c.execute('SELECT DISTINCT(objectid) FROM polygons')
        self.assertEqual(c.fetchall(), [(2, )])

    def test_deleteObject2(self):
        c = self.conn.cursor()
        backend_db.deleteObject(c, objectid=2)

        c.execute('SELECT objectid FROM objects')
        objectids = c.fetchall()
        self.assertEqual(objectids, [(1, ), (3, )], str(objectids))

        c.execute('SELECT DISTINCT(objectid) FROM properties')
        self.assertEqual(c.fetchall(), [(1, ), (3, )])

        c.execute('SELECT DISTINCT(objectid) FROM matches')
        self.assertEqual(c.fetchall(), [(1, )])

        c.execute('SELECT DISTINCT(objectid) FROM polygons')
        self.assertEqual(c.fetchall(), [])


class Test_updateObjectTransform_emptyDb(testing_utils.Test_emptyDb):

    def test_noPreviousTransform(self):
        c = self.conn.cursor()

        transform = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        backend_db.updateObjectTransform(c, 0, transform)

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
        backend_db.updateObjectTransform(c, 0, transform1)

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


class TestUpgradeV4toV5_Cars(testing_utils.Test_emptyDb):

    CARS_DB_V4_PATH = 'testdata/cars/micro1_v5.db'
    CARS_DB_V5_PATH = 'testdata/cars/micro1_v5.db'

    def setUp(self):
        self.temp_db_v4_path = tempfile.NamedTemporaryFile().name
        self.temp_db_v5_path = tempfile.NamedTemporaryFile().name

        shutil.copyfile(self.CARS_DB_V4_PATH, self.temp_db_v4_path)
        shutil.copyfile(self.CARS_DB_V5_PATH, self.temp_db_v5_path)

        self.conn = sqlite3.connect(self.temp_db_v4_path)
        backend_db.upgradeV4toV5(self.conn.cursor())

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
