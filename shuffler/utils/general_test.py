import os.path as op
import numpy as np
import shutil
import unittest
import tempfile
import progressbar
import nose
import sqlite3

from shuffler.backend import backend_db
from shuffler.utils import general as general_utils


class Test_takeSubpath(unittest.TestCase):
    def test_all(self):
        # Need to us os.path.join as opposed to '/' because must run on Windows.
        path1 = 'c'
        path2 = op.join('b', 'c')
        path3 = op.join('a', 'b', 'c')
        #
        self.assertEqual(general_utils.takeSubpath(path3, None), path3)
        self.assertEqual(general_utils.takeSubpath(path3, 4), path3)
        self.assertEqual(general_utils.takeSubpath(path3, 3), path3)
        self.assertEqual(general_utils.takeSubpath(path3, 2), path2)
        self.assertEqual(general_utils.takeSubpath(path3, 1), path1)
        with self.assertRaises(ValueError):
            general_utils.takeSubpath(path3, 0)
        with self.assertRaises(ValueError):
            general_utils.takeSubpath(path3, -1)
        with self.assertRaises(ValueError):
            general_utils.takeSubpath('', 1)


class Test_validateFileName(unittest.TestCase):
    def test_all(self):
        self.assertEqual(general_utils.validateFileName('abc.jpg'), 'abc.jpg')
        self.assertEqual(general_utils.validateFileName('ab c'), 'ab c')
        self.assertEqual(general_utils.validateFileName('ab!c'), 'ab_33_c')
        self.assertEqual(general_utils.validateFileName('ab/c'), 'ab_47_c')
        self.assertEqual(general_utils.validateFileName('ab\\c'), 'ab_92_c')


class Test_CopyWithBackup(unittest.TestCase):
    def setUp(self):
        self.work_dir = tempfile.mkdtemp()
        with open(op.join(self.work_dir, 'from.txt'), 'w') as f:
            f.write('from')

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_failsIfFileNotExists(self):
        from_path = op.join(self.work_dir, 'not_exists.txt')
        to_path = op.join(self.work_dir, 'to.txt')
        with self.assertRaises(FileNotFoundError):
            general_utils.copyWithBackup(from_path, to_path)

    def test_copiesWithoutNeedForBackup(self):
        from_path = op.join(self.work_dir, 'from.txt')
        to_path = op.join(self.work_dir, 'to.txt')
        general_utils.copyWithBackup(from_path, to_path)
        self.assertTrue(op.exists(from_path))
        self.assertTrue(op.exists(to_path))

    def test_copiesWithBackup(self):
        from_path = op.join(self.work_dir, 'from.txt')
        to_path = op.join(self.work_dir, 'to.txt')
        backup_path = op.join(self.work_dir, 'to.backup.txt')
        # Make an existing file.
        with open(to_path, 'w') as f:
            f.write('old')
        general_utils.copyWithBackup(from_path, to_path)
        # Make sure the contents are overwritten.
        self.assertTrue(op.exists(to_path))
        with open(to_path) as f:
            s = f.readline()
            self.assertEqual(s, 'from')
        # Make sure the old file was backed up.
        self.assertTrue(op.exists(backup_path))
        with open(backup_path) as f:
            s = f.readline()
            self.assertEqual(s, 'old')

    def test_copiestoItself(self):
        from_path = op.join(self.work_dir, 'from.txt')
        backup_path = op.join(self.work_dir, 'from.backup.txt')
        general_utils.copyWithBackup(from_path, from_path)
        # Make sure the contents are the same.
        self.assertTrue(op.exists(from_path))
        with open(from_path) as f:
            s = f.readline()
            self.assertEqual(s, 'from')
        # Make sure the old file was backed up.
        self.assertTrue(op.exists(backup_path))
        with open(backup_path) as f:
            s = f.readline()
            self.assertEqual(s, 'from')


class Test_bbox2polygon(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backend_db.createDb(self.conn)

    def test_general(self):
        c = self.conn.cursor()
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
        self.assertEqual(set(actual), set(expected))

        # Run on object 2, which has NO polygon entries.
        general_utils.bbox2polygon(c, objectid=2)
        c.execute('SELECT objectid,x,y FROM polygons ORDER BY objectid')
        actual = c.fetchall()
        # Should find the original entries for #1 and new entries for #2.
        expected = [(1, 40, 20), (1, 40.5, 30.5), (1, 50.5, 30),
                    (2, 20.5, 30.5), (2, 20.5, 41), (2, 41, 41), (2, 41, 30.5)]
        self.assertEqual(set(actual), set(expected))


class Test_polygon2bbox(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backend_db.createDb(self.conn)

    def test_general(self):
        c = self.conn.cursor()
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
        self.assertEqual(set(actual), set(expected))

        # Run on object 2. No changes.
        general_utils.polygon2bbox(c, objectid=2)
        c.execute('SELECT objectid,x1,y1,width,height FROM objects')
        actual = c.fetchall()
        expected = [(1, 40, 20, 10.5, 10.5), (2, 20.5, 30.5, 20.5, 10.5)]
        self.assertEqual(set(actual), set(expected))


class Test_polygon2mask(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backend_db.createDb(self.conn)

    def test_badInfo(self):
        c = self.conn.cursor()
        c = self.conn.cursor()
        c.execute('INSERT INTO objects(objectid) VALUES (1)')
        with self.assertRaises(RuntimeError):
            general_utils.polygons2mask(c, objectid=1)

    def test_general(self):
        c = self.conn.cursor()
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
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (100, 200))


class Test_getIntersectingObjects(unittest.TestCase):
    def test_empty(self):
        pairs_to_merge = general_utils.getIntersectingObjects([], [], 0.5)
        self.assertEqual(pairs_to_merge, [])

    def test_firstEmpty(self):
        objects2 = [(1, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects([], objects2,
                                                              0.5)
        self.assertEqual(pairs_to_merge, [])

    def test_identical(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.5)
        self.assertEqual(pairs_to_merge, [(1, 2)])

    def test_identical_sameId(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(1, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(objects1,
                                                              objects2,
                                                              0.5,
                                                              same_id_ok=False)
        self.assertEqual(pairs_to_merge, [])

    def test_nonIntersecting(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 20, 20, 40, 40, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.5)
        self.assertEqual(pairs_to_merge, [])

    def test_twoIntersecting(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 20, 20, 30, 30, 'name2', 1.),
                    (3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 40, 50, 60, 70, 'name4', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.1)
        self.assertEqual(pairs_to_merge, [(1, 3)])

    def test_twoAndTwoIntersecting(self):
        # #1.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 20, 20, 30, 30, 'name3', 1.),
                    (4, 'image', 10, 10, 30, 30, 'name4', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.1)
        self.assertEqual(set(pairs_to_merge), set([(1, 4), (2, 3)]))
        # #2.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 20, 20, 30, 30, 'name4', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.1)
        self.assertEqual(set(pairs_to_merge), set([(1, 3), (2, 4)]))
        # #3.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 0, 0, 30, 30, 'name4', 1.),
                    (5, 'image', 20, 20, 30, 30, 'name5', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.1)
        self.assertEqual(set(pairs_to_merge), set([(1, 3), (2, 5)]))


class Test_makeExportedImageName(unittest.TestCase):
    def setUp(self):
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_regular(self):
        tgt_path = general_utils.makeExportedImageName(self.work_dir,
                                                       'src_dir/filename')
        self.assertEqual(tgt_path, op.join(self.work_dir, 'filename'))

        # Make it exist.
        with open(tgt_path, 'w') as f:
            f.write('')

        with self.assertRaises(FileExistsError):
            general_utils.makeExportedImageName(self.work_dir,
                                                'src_dir/filename')

    def test_dirtreeLevelForName_eq2(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'my/fancy/filename', dirtree_level_for_name=2)
        self.assertEqual(tgt_path, 'tgt_dir/fancy_filename')

    def test_dirtreeLevelForName_eq3(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'my/fancy/filename', dirtree_level_for_name=3)
        self.assertEqual(tgt_path, 'tgt_dir/my_fancy_filename')

    def test_dirtreeLevelForName_eqALot(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'my/fancy/filename', dirtree_level_for_name=42)
        self.assertEqual(tgt_path, 'tgt_dir/my_fancy_filename')

    def test_fixInvalidImageNames(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'src_dir/file(?)name', fix_invalid_image_names=True)
        self.assertEqual(tgt_path, 'tgt_dir/file___name')


class Test_getMatchPolygons(unittest.TestCase):
    def test_empty(self):
        pairs = general_utils.getMatchPolygons([], [], 1.)
        self.assertEqual(pairs, [])

    def test_firstEmpty(self):
        objectid = 1
        polygons2 = [(1, objectid, 10, 30, 'name1')]
        pairs = general_utils.getMatchPolygons([], polygons2, 1.)
        self.assertEqual(pairs, [])

    def test_identical(self):
        ''' Identical points are not matched if when ignoring names. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1')]
        polygons2 = [(2, objectid, 10, 30, 'name2')]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., True)
        self.assertEqual(pairs, [(1, 2)])

    def test_identicalAndNameMatter(self):
        ''' Identical points are not matched if names differ. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1')]
        polygons2 = [(2, objectid, 10, 30, 'name2')]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., False)
        self.assertEqual(pairs, [])

    def test_someMatchingPointsAndNameIgnored(self):
        ''' Some points are matched, names are ignored. '''
        objectid = 1
        polygons1 = [(1, objectid, 100, 100, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        polygons2 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., True)
        self.assertEqual(pairs, [(2, 3)])

    def test_ambiguousPointsInPolygon1(self):
        ''' Expect an error if two points can be matched. Names are ignored. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        polygons2 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        # If there are matches of repeated points, then smth is wrong here.
        with self.assertRaises(ValueError):
            general_utils.getMatchPolygons(polygons1, polygons2, 1., True)

    def test_ambiguousPointsInPolygon2(self):
        ''' Expect an error if two points can be matched. Names are ignored. '''
        objectid = 1
        polygons1 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        polygons2 = [(1, objectid, 10, 30, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        # If there are matches of repeated points, then smth is wrong here.
        with self.assertRaises(ValueError):
            general_utils.getMatchPolygons(polygons1, polygons2, 1., True)

    def test_haveRepeatedPointsAndNameMatter(self):
        ''' Only the point with matching name is matched out of two points. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        polygons2 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., False)
        self.assertEqual(pairs, [(2, 3)])

    def test_haveNameIsNull(self):
        ''' Only the point with matching name is matched out of two points. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, None),
                     (2, objectid, 20, 40, 'name1'),
                     (3, objectid, 30, 50, 'name2')]
        polygons2 = [(4, objectid, 30, 50, 'name2'),
                     (5, objectid, 20, 40, 'name1'),
                     (6, objectid, 10, 30, None)]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., False)
        self.assertEqual(set(pairs), set([(1, 6), (2, 5), (3, 4)]))


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
