import os
import sqlite3
import shutil
import tempfile
import numpy as np
import pytest

from shuffler.backend import backend_db
from shuffler.utils import testing as testing_utils


class Test_FieldGetters_Cars(testing_utils.CarsDb):
    def test_objectField(self, c):
        c.execute('SELECT * FROM objects WHERE objectid=1')
        entry = c.fetchone()

        assert backend_db.objectField(entry, 'objectid') == 1
        assert backend_db.objectField(entry,
                                      'imagefile') == 'images/000000.jpg'
        assert backend_db.objectField(entry, 'x1') == 225.1
        assert backend_db.objectField(entry, 'y1') == 134.2
        assert backend_db.objectField(entry, 'width') == 356.3
        assert backend_db.objectField(entry, 'height') == 377.4
        assert backend_db.objectField(entry, 'name') == 'car'
        assert backend_db.objectField(entry,
                                      'score') == pytest.approx(0.606193)
        with pytest.raises(KeyError):
            backend_db.objectField(entry, 'dummy')

        # Multiple fields.
        assert backend_db.objectFields(entry, ['objectid', 'x1']) == [1, 225.1]
        with pytest.raises(KeyError):
            backend_db.objectFields(entry, ['x1', 'dummy'])

    def test_imageField(self, c):
        c.execute('SELECT * FROM images WHERE imagefile="images/000000.jpg"')
        entry = c.fetchone()

        assert backend_db.imageField(entry, 'imagefile') == 'images/000000.jpg'
        assert backend_db.imageField(entry, 'width') == 800
        assert backend_db.imageField(entry, 'height') == 700
        assert backend_db.imageField(entry, 'maskfile') == 'masks/000000.png'
        assert backend_db.imageField(
            entry, 'timestamp') == '2018-09-24 12:22:48.534685'
        assert backend_db.imageField(entry, 'name') is None
        assert backend_db.imageField(entry, 'score') is None
        with pytest.raises(KeyError):
            backend_db.imageField(entry, 'dummy')

        # Multiple fields.
        assert backend_db.imageFields(entry, ['name', 'width']) == [None, 800]
        with pytest.raises(KeyError):
            backend_db.imageFields(entry, ['name', 'dummy'])

    def test_polygonField(self, c):
        c.execute('SELECT * FROM polygons WHERE id=1')
        entry = c.fetchone()

        assert backend_db.polygonField(entry, 'id') == 1
        assert backend_db.polygonField(entry, 'objectid') == 2
        assert backend_db.polygonField(entry, 'x') == 97.1
        assert backend_db.polygonField(entry, 'y') == 296.0
        assert backend_db.polygonField(entry, 'name') is None
        with pytest.raises(KeyError):
            backend_db.polygonField(entry, 'dummy')

        # Multiple fields.
        assert backend_db.polygonFields(entry, ['id', 'x']) == [1, 97.1]
        with pytest.raises(KeyError):
            backend_db.polygonFields(entry, ['id', 'dummy'])

    def test_getColumnsInTable(self, c):
        result = backend_db.getColumnsInTable(c, 'objects')
        assert result == [
            'objectid', 'imagefile', 'x1', 'y1', 'width', 'height', 'name',
            'score'
        ]


class Test_DeleteImage_Cars(testing_utils.CarsDb):
    def test_delete_imagefile_nonexistent(self, c):
        with pytest.raises(KeyError):
            backend_db.deleteImage(c, imagefile='not_existent')

    def test_delete_imagefile000000(self, c):
        backend_db.deleteImage(c, imagefile='images/000000.jpg')

        c.execute('SELECT imagefile FROM images')
        imagefiles = c.fetchall()
        assert imagefiles == [('images/000001.jpg', ), ('images/000002.jpg', )]

        c.execute('SELECT objectid FROM objects')
        objectids = c.fetchall()
        assert objectids == [(2, ), (3, )], str(objectids)

        c.execute('SELECT DISTINCT(objectid) FROM properties')
        assert c.fetchall() == [(2, ), (3, )]

        c.execute('SELECT DISTINCT(objectid) FROM matches')
        assert c.fetchall() == [(2, )]

        c.execute('SELECT DISTINCT(objectid) FROM polygons')
        assert c.fetchall() == [(2, )]

    def test_delete_imagefile000001(self, c):
        backend_db.deleteImage(c, imagefile='images/000001.jpg')

        c.execute('SELECT imagefile FROM images')
        imagefiles = c.fetchall()
        assert imagefiles == [('images/000000.jpg', ), ('images/000002.jpg', )]

        c.execute('SELECT objectid FROM objects')
        objectids = c.fetchall()
        assert objectids, [(1, )] == str(objectids)

        c.execute('SELECT DISTINCT(objectid) FROM properties')
        assert c.fetchall() == [(1, )]

        c.execute('SELECT DISTINCT(objectid) FROM matches')
        assert c.fetchall() == [(1, )]

        c.execute('SELECT DISTINCT(objectid) FROM polygons')
        assert c.fetchall() == []


class Test_DeleteObject_Cars(testing_utils.CarsDb):
    def test_delete_object0(self, c):
        with pytest.raises(KeyError):
            backend_db.deleteObject(c, objectid=0)

    def test_delete_object1(self, c):
        backend_db.deleteObject(c, objectid=1)

        c.execute('SELECT objectid FROM objects')
        objectids = c.fetchall()
        assert objectids == [(2, ), (3, )]

        c.execute('SELECT DISTINCT(objectid) FROM properties')
        assert c.fetchall() == [(2, ), (3, )]

        c.execute('SELECT DISTINCT(objectid) FROM matches')
        assert c.fetchall() == [(2, )]

        c.execute('SELECT DISTINCT(objectid) FROM polygons')
        assert c.fetchall() == [(2, )]

    def test_delete_object2(self, c):
        backend_db.deleteObject(c, objectid=2)

        c.execute('SELECT objectid FROM objects')
        objectids = c.fetchall()
        assert objectids == [(1, ), (3, )]

        c.execute('SELECT DISTINCT(objectid) FROM properties')
        assert c.fetchall() == [(1, ), (3, )]

        c.execute('SELECT DISTINCT(objectid) FROM matches')
        assert c.fetchall() == [(1, )]

        c.execute('SELECT DISTINCT(objectid) FROM polygons')
        assert c.fetchall() == []


class Test_UpdateObjectTransform_EmptyDb(testing_utils.EmptyDb):
    def test_no_previous_transform(self, c):
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
        assert (len(ky), len(kx), len(by), len(bx)) == (1, 1, 1, 1)
        assert float(ky[0][0]) == 1.
        assert float(kx[0][0]) == 1.
        assert float(by[0][0]) == 0.
        assert float(bx[0][0]) == 0.

    def test_with_previous_transform(self, c):
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
        assert (len(ky), len(kx), len(by), len(bx)) == (1, 1, 1, 1)
        transform2 = np.array([[float(ky[0][0]), 0.,
                                float(by[0][0])],
                               [0., float(kx[0][0]),
                                float(bx[0][0])], [0., 0., 1.]])

        np.testing.assert_array_equal(np.matmul(transform1, transform0),
                                      transform2)


class Test_CheckSameMaskfileHasSameDims(testing_utils.EmptyDb):
    def test_good(self, c):
        c.execute('INSERT INTO images(imagefile,maskfile,width,height) VALUES '
                  '("a",  "x", 60, 40), '
                  '("b",  "x", 60, 40), '
                  '(NULL, "x", 60, 40), '
                  '("c",  "y", 200, 200), '
                  '("d", NULL, 300, 500), '
                  '("e", NULL, 400, 600) ')
        backend_db.checkSameMaskfileHasSameDims(c)

    def test_bad1(self, c):
        c.execute('INSERT INTO images(imagefile,maskfile,width,height) VALUES '
                  '("a",  "x", 60, 40), '
                  '("b",  "x", 100, 100) ')
        with pytest.raises(ValueError):
            backend_db.checkSameMaskfileHasSameDims(c)

    def test_bad2(self, c):
        c.execute('INSERT INTO images(imagefile,maskfile,width,height) VALUES '
                  '("a",  "x", 60, 40), '
                  '("NULL",  "x", 100, 100) ')
        with pytest.raises(ValueError):
            backend_db.checkSameMaskfileHasSameDims(c)


class Test_UpgradeV4toV5:

    CARS_DB_V4_PATH = 'testdata/cars/micro1_v5.db'  # Purposedly v5.
    CARS_DB_V5_PATH = 'testdata/cars/micro1_v5.db'

    @pytest.fixture()
    def c_and_temp_db_v5_path(self):
        temp_db_v4_path = tempfile.NamedTemporaryFile().name
        temp_db_v5_path = tempfile.NamedTemporaryFile().name

        shutil.copyfile(self.CARS_DB_V4_PATH, temp_db_v4_path)
        shutil.copyfile(self.CARS_DB_V5_PATH, temp_db_v5_path)

        conn = sqlite3.connect(temp_db_v4_path)
        backend_db.upgradeV4toV5(conn.cursor())
        yield (conn.cursor(), temp_db_v5_path)
        conn.close()
        if os.path.exists(temp_db_v4_path):
            os.remove(temp_db_v4_path)
        if os.path.exists(temp_db_v5_path):
            os.remove(temp_db_v5_path)

    def test_table_contents(self, c_and_temp_db_v5_path):
        ''' Compare the contents of "table" between "main" and "gt" schema names. '''

        c, temp_db_v5_path = c_and_temp_db_v5_path

        c.execute('ATTACH "%s" AS gt' % temp_db_v5_path)

        for table in [
                'images', 'objects', 'matches', 'polygons', 'properties'
        ]:
            c.execute('SELECT * FROM %s' % table)
            entries_main = c.fetchall()
            c.execute('SELECT * FROM gt.%s' % table)
            entries_gt = c.fetchall()
            assert entries_main == entries_gt

    def test_index_names(self, c_and_temp_db_v5_path):
        c, temp_db_v5_path = c_and_temp_db_v5_path

        c.execute('SELECT name FROM sqlite_master WHERE type == "index"')
        entries_main = c.fetchall()
        c.execute('ATTACH "%s" AS gt' % temp_db_v5_path)
        c.execute('SELECT name FROM sqlite_master WHERE type == "index"')
        entries_gt = c.fetchall()
        assert entries_main == entries_gt

        # TODO: Test that indexes actually have the same info.
