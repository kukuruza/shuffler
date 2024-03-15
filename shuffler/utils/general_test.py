import pytest
import os.path as op
import numpy as np
import shutil
import tempfile
import sqlite3

from shuffler.backend import backend_db
from shuffler.utils import general as general_utils
from shuffler.utils import boxes as boxes_utils


class Test_takeSubpath:
    def test_all(self):
        # Need to us os.path.join as opposed to '/' because must run on Windows.
        path1 = 'c'
        path2 = op.join('b', 'c')
        path3 = op.join('a', 'b', 'c')
        #
        assert general_utils.takeSubpath(path3, None) == path3
        assert general_utils.takeSubpath(path3, 4) == path3
        assert general_utils.takeSubpath(path3, 3) == path3
        assert general_utils.takeSubpath(path3, 2) == path2
        assert general_utils.takeSubpath(path3, 1) == path1
        with pytest.raises(ValueError):
            general_utils.takeSubpath(path3, 0)
        with pytest.raises(ValueError):
            general_utils.takeSubpath(path3, -1)
        with pytest.raises(ValueError):
            general_utils.takeSubpath('', 1)


class Test_validateFileName:
    def test_all(self):
        assert general_utils.validateFileName('abc.jpg') == 'abc.jpg'
        assert general_utils.validateFileName('ab c') == 'ab c'
        assert general_utils.validateFileName('ab!c') == 'ab_33_c'
        assert general_utils.validateFileName('ab/c') == 'ab_47_c'
        assert general_utils.validateFileName('ab\\c') == 'ab_92_c'


class Test_copyWithBackup:
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        with open(op.join(work_dir, 'from.txt'), 'w') as f:
            f.write('from')
        yield work_dir
        shutil.rmtree(work_dir)

    def test_fails_if_file_not_exists(self, work_dir):
        from_path = op.join(work_dir, 'not_exists.txt')
        to_path = op.join(work_dir, 'to.txt')
        with pytest.raises(FileNotFoundError):
            general_utils.copyWithBackup(from_path, to_path)

    def test_copies_without_need_for_backup(self, work_dir):
        from_path = op.join(work_dir, 'from.txt')
        to_path = op.join(work_dir, 'to.txt')
        general_utils.copyWithBackup(from_path, to_path)
        assert op.exists(from_path)
        assert op.exists(to_path)

    def test_copies_with_backup(self, work_dir):
        from_path = op.join(work_dir, 'from.txt')
        to_path = op.join(work_dir, 'to.txt')
        backup_path = op.join(work_dir, 'to.backup.txt')
        # Make an existing file.
        with open(to_path, 'w') as f:
            f.write('old')
        general_utils.copyWithBackup(from_path, to_path)
        # Make sure the contents are overwritten.
        assert op.exists(to_path)
        with open(to_path) as f:
            s = f.readline()
            assert s == 'from'
        # Make sure the old file was backed up.
        assert op.exists(backup_path)
        with open(backup_path) as f:
            s = f.readline()
            assert s == 'old'

    def test_copies_to_itself(self, work_dir):
        from_path = op.join(work_dir, 'from.txt')
        backup_path = op.join(work_dir, 'from.backup.txt')
        general_utils.copyWithBackup(from_path, from_path)
        # Make sure the contents are the same.
        assert op.exists(from_path)
        with open(from_path) as f:
            s = f.readline()
            assert s == 'from'
        # Make sure the old file was backed up.
        assert op.exists(backup_path)
        with open(backup_path) as f:
            s = f.readline()
            assert s == 'from'


class Test_bbox2polygon:
    @pytest.fixture()
    def c(self):
        conn = sqlite3.connect(':memory:')
        backend_db.createDb(conn)
        yield conn.cursor()

    def test_general(self, c):
        # Two objects.
        c.execute('INSERT INTO objects(objectid,x1,y1,width,height) '
                  'VALUES (1, 40, 20, 10, 10), (2, 20.5, 30.5, 20.5, 10.5)')
        # Object 1 has a polygon.
        c.execute('INSERT INTO polygons(objectid,x,y) '
                  'VALUES (1, 40, 20), (1, 40.5, 30.5), (1, 50.5, 30)')

        # Run on object 1, which already has polygon entries.
        general_utils.bbox2polygon(c, objectid=1)
        c.execute('SELECT objectid,x,y FROM polygons ORDER BY objectid')
        actual = c.fetchall()
        expected = [(1, 40, 20), (1, 40.5, 30.5), (1, 50.5, 30)]
        assert set(actual) == set(expected)

        # Run on object 2, which has NO polygon entries.
        general_utils.bbox2polygon(c, objectid=2)
        c.execute('SELECT objectid,x,y FROM polygons ORDER BY objectid')
        actual = c.fetchall()
        # Should find the original entries for #1 and new entries for #2.
        expected = [(1, 40, 20), (1, 40.5, 30.5), (1, 50.5, 30),
                    (2, 20.5, 30.5), (2, 20.5, 41), (2, 41, 41), (2, 41, 30.5)]
        assert set(actual) == set(expected)


class Test_polygon2bbox:
    @pytest.fixture()
    def c(self):
        conn = sqlite3.connect(':memory:')
        backend_db.createDb(conn)
        yield conn.cursor()

    def test_general(self, c):
        # Two objects.
        c.execute('INSERT INTO objects(objectid,x1,y1,width,height) '
                  'VALUES (1, 40, 20, 10, 10), (2, 20.5, 30.5, 20.5, 10.5)')
        # Object 1 has a polygon.
        c.execute('INSERT INTO polygons(objectid,x,y) '
                  'VALUES (1, 40, 20), (1, 40.5, 30.5), (1, 50.5, 30)')

        # Run on object 1. Bbox for object 1 should change.
        general_utils.polygon2bbox(c, objectid=1)
        c.execute('SELECT objectid,x1,y1,width,height FROM objects')
        actual = c.fetchall()
        expected = [(1, 40, 20, 10.5, 10.5), (2, 20.5, 30.5, 20.5, 10.5)]
        assert set(actual) == set(expected)

        # Run on object 2. No changes.
        general_utils.polygon2bbox(c, objectid=2)
        c.execute('SELECT objectid,x1,y1,width,height FROM objects')
        actual = c.fetchall()
        expected = [(1, 40, 20, 10.5, 10.5), (2, 20.5, 30.5, 20.5, 10.5)]
        assert set(actual) == set(expected)


class Test_Polygon2mask:
    @pytest.fixture()
    def c(self):
        conn = sqlite3.connect(':memory:')
        backend_db.createDb(conn)
        yield conn.cursor()

    def test_bad_info(self, c):
        c.execute('INSERT INTO objects(objectid) VALUES (1)')
        with pytest.raises(RuntimeError):
            general_utils.polygons2mask(c, objectid=1)

    def test_general(self, c):
        # One image of size 200x100.
        c.execute('INSERT INTO images(imagefile,width,height) '
                  'VALUES ("image0",200,100)')
        # Two objects.
        c.execute('INSERT INTO objects(imagefile,objectid) '
                  'VALUES ("image0",1), ("image0",2)')
        # Object 1 has a polygon.
        c.execute(
            'INSERT INTO polygons(objectid,x,y) '
            'VALUES (1, 40, 20), (1, 40.1, 30.1), (1, 50.1, 30), (1, 50, 20)')

        # Run on object 1. Bbox for object 1 should change.
        mask = general_utils.polygons2mask(c, objectid=1)
        assert mask.dtype == np.uint8
        assert mask.shape == (100, 200)


class Test_getPolygonsByObject:
    @pytest.fixture()
    def c(self):
        conn = sqlite3.connect(':memory:')
        backend_db.createDb(conn)
        yield conn.cursor()

    def test_general(self, c):
        # Two objects.
        c.execute('INSERT INTO objects(objectid,x1,y1,width,height) '
                  'VALUES (1, 40, 20, 10, 10), (2, 20.5, 30.5, 20.5, 10.5)')
        # Object 1 has a polygon.
        c.execute('INSERT INTO polygons(objectid,x,y) '
                  'VALUES (1, 40, 20), (1, 40.5, 30.5), (1, 50.5, 30)')

        expected = {
            1: [(20, 40), (30.5, 40.5), (30, 50.5)],
            2: [(30.5, 20.5), (41.0, 20.5), (41.0, 41.0), (30.5, 41.0)]
        }

        c.execute('SELECT * FROM objects')
        objects = c.fetchall()
        assert general_utils.getPolygonsByObject(c, objects) == expected


class Test_getIntersectingObjects:
    def _getPolygonsByObjectViaBbox(self, objects):
        return {
            backend_db.objectField(object_, 'objectid'):
            boxes_utils.box2polygon(backend_db.objectField(object_, 'bbox'))
            for object_ in objects
        }

    def test_empty(self):
        pairs_to_merge = general_utils.getIntersectingObjects({}, {}, 0.5)
        assert pairs_to_merge == []

    def test_first_empty(self):
        objects2 = [(1, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            {}, self._getPolygonsByObjectViaBbox(objects2), 0.5)
        assert pairs_to_merge == []

    def test_identical(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            self._getPolygonsByObjectViaBbox(objects1),
            self._getPolygonsByObjectViaBbox(objects2), 0.5)
        assert pairs_to_merge == [(1, 2)]

    def test_within_itself(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            self._getPolygonsByObjectViaBbox(objects1), None, 0.1)
        assert pairs_to_merge == [(1, 2)]

    def test_non_intersecting(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 20, 20, 40, 40, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            self._getPolygonsByObjectViaBbox(objects1),
            self._getPolygonsByObjectViaBbox(objects2), 0.5)
        assert pairs_to_merge == []

    def test_two_intersecting(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 20, 20, 30, 30, 'name2', 1.),
                    (3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 40, 50, 60, 70, 'name4', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            self._getPolygonsByObjectViaBbox(objects1),
            self._getPolygonsByObjectViaBbox(objects2), 0.1)
        assert pairs_to_merge == [(1, 3)]

    def test_two_and_two_intersecting(self):
        # #1.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 20, 20, 30, 30, 'name3', 1.),
                    (4, 'image', 10, 10, 30, 30, 'name4', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            self._getPolygonsByObjectViaBbox(objects1),
            self._getPolygonsByObjectViaBbox(objects2), 0.1)
        assert set(pairs_to_merge) == set([(1, 4), (2, 3)])
        # #2.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 20, 20, 30, 30, 'name4', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            self._getPolygonsByObjectViaBbox(objects1),
            self._getPolygonsByObjectViaBbox(objects2), 0.1)
        assert set(pairs_to_merge) == set([(1, 3), (2, 4)])
        # #3.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 0, 0, 30, 30, 'name4', 1.),
                    (5, 'image', 20, 20, 30, 30, 'name5', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            self._getPolygonsByObjectViaBbox(objects1),
            self._getPolygonsByObjectViaBbox(objects2), 0.1)
        assert set(pairs_to_merge) == set([(1, 3), (2, 5)])


class Test_MakeExportedImageName:
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        yield work_dir
        shutil.rmtree(work_dir)

    def test_regular(self, work_dir):
        tgt_path = general_utils.makeExportedImageName(work_dir,
                                                       'src_dir/filename')
        assert tgt_path == op.join(work_dir, 'filename')

        # Make it exist.
        with open(tgt_path, 'w') as f:
            f.write('')

        with pytest.raises(FileExistsError):
            general_utils.makeExportedImageName(work_dir, 'src_dir/filename')

    def test_dirtree_level_for_name_eq2(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'my/fancy/filename', dirtree_level_for_name=2)
        assert tgt_path == 'tgt_dir/fancy_filename'

    def test_dirtree_level_for_name_eq3(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'my/fancy/filename', dirtree_level_for_name=3)
        assert tgt_path == 'tgt_dir/my_fancy_filename'

    def test_dirtree_level_for_name_eqALot(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'my/fancy/filename', dirtree_level_for_name=42)
        assert tgt_path == 'tgt_dir/my_fancy_filename'

    def test_fix_invalid_image_names(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'src_dir/file(?)name', fix_invalid_image_names=True)
        assert tgt_path == 'tgt_dir/file___name'


class Test_MatchPolygonPoints:
    def test_empty(self):
        pairs = general_utils.matchPolygonPoints([], [], 1.)
        assert pairs == []

    def test_first_empty(self):
        objectid = 1
        polygons2 = [(1, objectid, 10, 30, 'name1')]
        pairs = general_utils.matchPolygonPoints([], polygons2, 1.)
        assert pairs == []

    def test_identical(self):
        ''' Identical points are not matched if when ignoring names. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1')]
        polygons2 = [(2, objectid, 10, 30, 'name2')]
        pairs = general_utils.matchPolygonPoints(polygons1, polygons2, 1.,
                                                 True)
        assert pairs == [(1, 2)]

    def test_identical_and_name_matter(self):
        ''' Identical points are not matched if names differ. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1')]
        polygons2 = [(2, objectid, 10, 30, 'name2')]
        pairs = general_utils.matchPolygonPoints(polygons1, polygons2, 1.,
                                                 False)
        assert pairs == []

    def test_some_matching_points_and_name_ignored(self):
        ''' Some points are matched, names are ignored. '''
        objectid = 1
        polygons1 = [(1, objectid, 100, 100, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        polygons2 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        pairs = general_utils.matchPolygonPoints(polygons1, polygons2, 1.,
                                                 True)
        assert pairs == [(2, 3)]

    def test_ambiguous_points_in_polygon1(self):
        ''' Expect an error if two points can be matched. Names are ignored. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        polygons2 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        # If there are matches of repeated points, then smth is wrong here.
        with pytest.raises(ValueError):
            general_utils.matchPolygonPoints(polygons1, polygons2, 1., True)

    def test_ambiguous_points_in_polygon2(self):
        ''' Expect an error if two points can be matched. Names are ignored. '''
        objectid = 1
        polygons1 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        polygons2 = [(1, objectid, 10, 30, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        # If there are matches of repeated points, then smth is wrong here.
        with pytest.raises(ValueError):
            general_utils.matchPolygonPoints(polygons1, polygons2, 1., True)

    def test_have_repeated_points_and_name_matter(self):
        ''' Only the point with matching name is matched out of two points. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        polygons2 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        pairs = general_utils.matchPolygonPoints(polygons1, polygons2, 1.,
                                                 False)
        assert pairs == [(2, 3)]

    def test_have_name_is_null(self):
        ''' Only the point with matching name is matched out of two points. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, None),
                     (2, objectid, 20, 40, 'name1'),
                     (3, objectid, 30, 50, 'name2')]
        polygons2 = [(4, objectid, 30, 50, 'name2'),
                     (5, objectid, 20, 40, 'name1'),
                     (6, objectid, 10, 30, None)]
        pairs = general_utils.matchPolygonPoints(polygons1, polygons2, 1.,
                                                 False)
        assert set(pairs) == set([(1, 6), (2, 5), (3, 4)])
