import os, os.path as op
import logging
import sqlite3
import shutil
import progressbar
import unittest
import argparse
import tempfile
import nose

from lib.backend import backendDb
from lib.subcommands import dbModify
from lib.utils import testUtils


class Test_bboxesToPolygons_carsDb(testUtils.Test_carsDb):
    def test_general(self):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR)
        dbModify.bboxesToPolygons(c, args)
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 2, 0])
        self.assert_polygons_count_by_object(c, expected=[4, 5, 4])


class Test_revertObjectTransforms_SyntheticDb(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("image0",0,45,25,10,10)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (0,45,25)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (0,55,25)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (0,55,35)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (0,45,35)')
        # transform = [[2., 0.,   5.]
        #              [0., 0.5, -5.]]
        c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (0,"kx","2.")')
        c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (0,"ky","0.5")')
        c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (0,"bx","5")')
        c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (0,"by","-5.")')
        # The original bbox (x1, y1, width, height).
        self.original_bbox_gt = (20, 60, 5, 20)
        # The original polygon [(x, y)] * N.
        self.original_polygon_gt = [(20, 60), (25, 60), (25, 80), (20, 80)]

    def test_general(self):
        c = self.conn.cursor()
        args = argparse.Namespace(rootdir='.')
        dbModify.revertObjectTransforms(c, args)
        # Check bbox.
        c.execute('SELECT x1,y1,width,height FROM objects')
        original_bboxes = c.fetchall()
        self.assertEqual(len(original_bboxes), 1)
        original_bbox = original_bboxes[0]
        self.assertEqual(self.original_bbox_gt, original_bbox)
        # Check polygons.
        c.execute('SELECT x,y FROM polygons')
        original_polygon = c.fetchall()
        self.assertEqual(len(original_polygon), 4)
        self.assertEqual(self.original_polygon_gt, original_polygon)


class Test_moveRootdir_SyntheticDb(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("a/b")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("a/b",0,10,20,30,40)')

    def _assertImagesAndObjectsConsistency(self, c):
        c.execute('SELECT COUNT(1) FROM images')
        num_images = c.fetchone()[0]
        c.execute('SELECT COUNT(1) FROM images i JOIN objects o '
                  'WHERE i.imagefile = o.imagefile')
        num_same = c.fetchone()[0]
        self.assertEqual(num_images, num_same)

    def _assertResult(self, rootdir, newrootdir, expected):
        c = self.conn.cursor()
        args = argparse.Namespace(rootdir=rootdir,
                                  newrootdir=newrootdir,
                                  verify_paths=False)
        dbModify.moveRootdir(c, args)
        self._assertImagesAndObjectsConsistency(c)
        c.execute('SELECT imagefile FROM images')
        imagefile, = c.fetchone()
        self.assertEqual(imagefile, expected)

    def test1(self):
        self._assertResult(rootdir='.', newrootdir='.', expected='a/b')

    def test2(self):
        self._assertResult(rootdir='.', newrootdir='a', expected='b')

    def test3(self):
        self._assertResult(rootdir='.', newrootdir='c', expected='../a/b')


class Test_propertyToObjectsField_SyntheticDb(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("a"), ("b"), ("c")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,name,score) '
                  'VALUES ("a",0,10,"cat",0.1), ("b",1,20,"dog",0.2), '
                  '("c",2,30,"pig",0.3)')
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"color","gray"), (1,"breed","poodle")')
        c.execute('INSERT INTO polygons(objectid,x) VALUES (0,25), (1,35)')
        c.execute('INSERT INTO matches(objectid,match) VALUES (0,0), (1,0)')

    def testTrivial(self):
        ''' Test on the empty database. '''
        self.conn.close()
        conn = sqlite3.connect(':memory:')
        backendDb.createDb(conn)
        c = conn.cursor()
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","dummy")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='name',
                                  properties_key='newval')
        dbModify.propertyToObjectsField(c, args)

    def testBadField(self):
        c = self.conn.cursor()
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='bad_field',
                                  properties_key='newval')
        with self.assertRaises(ValueError):
            dbModify.propertyToObjectsField(c, args)

    def testAbsentKey(self):
        c = self.conn.cursor()
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='objectid',
                                  properties_key='absent_key')
        with self.assertRaises(ValueError):
            dbModify.propertyToObjectsField(c, args)

    def testObjectidField(self):
        c = self.conn.cursor()
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","2"), (2,"newval","6")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='objectid',
                                  properties_key='newval')
        dbModify.propertyToObjectsField(c, args)

        # Verify the "objects" table.
        c.execute('SELECT imagefile,objectid FROM objects')
        expected = [("a", 2), ("b", 1), ("c", 6)]
        self.assertEqual(set(c.fetchall()), set(expected))

        # Verify the "polygons" table (the newval 0 is replaced with 2).
        c.execute('SELECT objectid,x FROM polygons')
        expected = [(2, 25), (1, 35)]
        self.assertEqual(set(c.fetchall()), set(expected))

        # Verify the "matches" table (the newval 0 is replaced with 2).
        c.execute('SELECT objectid,match FROM matches')
        expected = [(2, 0), (1, 0)]
        self.assertEqual(set(c.fetchall()), set(expected))

        # Verify the "properties" table (the newval 0 is replaced with 2).
        c.execute('SELECT objectid,key,value FROM properties')
        expected = [(2, "color", "gray"), (1, "breed", "poodle"),
                    (2, "newval", "2"), (6, "newval", "6")]
        self.assertEqual(set(c.fetchall()), set(expected))

    def testObjectidField_NonUniqueValues(self):
        c = self.conn.cursor()
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","2"), (1,"newval","2")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='objectid',
                                  properties_key='newval')
        with self.assertRaises(Exception):
            dbModify.propertyToObjectsField(c, args)

    def testObjectidField_ValueMatchesNotUpdatedEntry(self):
        c = self.conn.cursor()
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","1")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='objectid',
                                  properties_key='newval')
        with self.assertRaises(Exception):
            dbModify.propertyToObjectsField(c, args)

    def testNameField(self):
        c = self.conn.cursor()
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","sheep")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='name',
                                  properties_key='newval')
        dbModify.propertyToObjectsField(c, args)

        # Verify the "objects" table.
        c.execute('SELECT objectid,name FROM objects')
        expected = [(0, "sheep"), (1, "dog"), (2, "pig")]
        self.assertEqual(set(c.fetchall()), set(expected))

    def testX1Field(self):
        c = self.conn.cursor()
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","50")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='x1',
                                  properties_key='newval')
        dbModify.propertyToObjectsField(c, args)

        # Verify the "objects" table.
        c.execute('SELECT objectid,x1 FROM objects')
        expected = [(0, 50), (1, 20), (2, 30)]
        self.assertEqual(set(c.fetchall()), set(expected))

    def testScoreField(self):
        c = self.conn.cursor()
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","0.5")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='score',
                                  properties_key='newval')
        dbModify.propertyToObjectsField(c, args)

        # Verify the "objects" table.
        c.execute('SELECT objectid,score FROM objects')
        expected = [(0, 0.5), (1, 0.2), (2, 0.3)]
        self.assertEqual(set(c.fetchall()), set(expected))


class Test_syncPolygonIdsWithDb_SyntheticDb(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)

        self.ref_db_path = tempfile.NamedTemporaryFile().name
        self.ref_conn = sqlite3.connect(self.ref_db_path)
        backendDb.createDb(self.ref_conn)

    def tearDown(self):
        self.conn.close()
        if op.exists(self.ref_db_path):
            os.remove(self.ref_db_path)

    def _vals2str(self, vals):
        '''
        Makes a string from INSERT values.
        Args:
          vals:  a list of tuples, e.g. [(0, 1), (1, 2)].
        Returns:
          a string, e.g. '(0, 1), (1, 2)'.
        '''
        vals_str = []
        for val in vals:
            val_str = ','.join(
                ['"%s"' % x if x is not None else 'NULL' for x in val])
            vals_str.append(val_str)
        s = ', '.join(['(%s)' % x for x in vals_str])
        # logging.debug('_vals2str build string: %s', s)
        return s

    def _insertPolygonsValue(self, vals, vals_ref):
        ''' Insert values into 'polygons' table of the active and ref dbs.
        Args:
          vals, vals_ref:  a tuple with 5 numbers (id,objectid,x,y,name).
        '''
        c = self.conn.cursor()
        c_ref = self.ref_conn.cursor()
        s = 'polygons(id,objectid,x,y,name)'
        if len(vals):
            c.execute('INSERT INTO %s VALUES %s' % (s, self._vals2str(vals)))
        if len(vals_ref):
            c_ref.execute('INSERT INTO %s VALUES %s' %
                          (s, self._vals2str(vals_ref)))
        self.ref_conn.commit()
        self.ref_conn.close()

    def test_empty(self):
        vals_ref = [(1, 1, 10, 20, 'name')]
        self._insertPolygonsValue([], vals_ref)
        c = self.conn.cursor()
        args = argparse.Namespace(ref_db_file=self.ref_db_path,
                                  epsilon=1.,
                                  ignore_name=False)
        dbModify.syncPolygonIdsWithDb(c, args)
        c.execute('SELECT id,objectid,x,y,name FROM polygons')
        self.assertEqual(c.fetchall(), [])

    def test_noUpdateBecauseOfDifferentObject(self):
        ''' No update is expected because the objects mismatch. '''
        vals = [(1, 1, 10, 20, 'name1'), (2, 1, 10, 20, 'name2')]
        vals_ref = [(1, 2, 10, 20, 'name1'), (2, 2, 10, 20, 'name2')]
        self._insertPolygonsValue(vals, vals_ref)
        c = self.conn.cursor()
        args = argparse.Namespace(ref_db_file=self.ref_db_path,
                                  epsilon=1.,
                                  ignore_name=False)
        dbModify.syncPolygonIdsWithDb(c, args)

        # Not checking ids here, because unmatched polygons points get new ids.
        vals_expected = [(1, 10, 20, 'name1'), (1, 10, 20, 'name2')]
        c.execute('SELECT objectid,x,y,name FROM polygons')
        self.assertEqual(c.fetchall(), vals_expected)

    def test_allMatch_ignoreName(self):
        ''' 
        All points match. Objectids is different. Ignore name. 
        '''
        # Will need to reverse ids.
        vals = [(4, 2, 10.5, 0.5, None), (3, 1, 20.5, 0.5, None),
                (2, 1, 10.5, 0.5, None), (1, 2, 20.5, 0.5, None)]
        vals_ref = [(1, 2, 10, 0, 'name1'), (2, 1, 20, 0, 'name2'),
                    (3, 1, 10, 0, 'name3'), (4, 2, 20, 0, 'name4')]
        self._insertPolygonsValue(vals, vals_ref)
        c = self.conn.cursor()
        args = argparse.Namespace(ref_db_file=self.ref_db_path,
                                  epsilon=1.,
                                  ignore_name=True)
        dbModify.syncPolygonIdsWithDb(c, args)

        vals_expected = [(1, 2, 10.5, 0.5, None), (2, 1, 20.5, 0.5, None),
                         (3, 1, 10.5, 0.5, None), (4, 2, 20.5, 0.5, None)]
        c.execute('SELECT id,objectid,x,y,name FROM polygons ORDER BY id ASC')
        self.assertEqual(c.fetchall(), vals_expected)

    def test_allMatch_matchName(self):
        ''' 
        All points match. Objectids and coordinates is the same. Use name.
        '''
        # Will need to reverse ids.
        vals = [(4, 1, 10.5, 0.5, 'name4'), (3, 1, 10.5, 0.5, 'name3'),
                (2, 1, 10.5, 0.5, None), (1, 1, 10.5, 0.5, 'name1')]
        vals_ref = [(1, 1, 10, 0, 'name1'), (2, 1, 10, 0, None),
                    (3, 1, 10, 0, 'name3'), (4, 1, 10, 0, 'name4')]
        self._insertPolygonsValue(vals, vals_ref)
        c = self.conn.cursor()
        args = argparse.Namespace(ref_db_file=self.ref_db_path,
                                  epsilon=1.,
                                  ignore_name=False)
        dbModify.syncPolygonIdsWithDb(c, args)

        vals_expected = [(1, 1, 10.5, 0.5, 'name1'), (2, 1, 10.5, 0.5, None),
                         (3, 1, 10.5, 0.5, 'name3'),
                         (4, 1, 10.5, 0.5, 'name4')]
        c.execute('SELECT id,objectid,x,y,name FROM polygons ORDER BY id ASC')
        self.assertEqual(c.fetchall(), vals_expected)


class Test_syncRoundedCoordinatesWithDb_SyntheticDb(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)

        self.ref_db_path = tempfile.NamedTemporaryFile().name
        self.ref_conn = sqlite3.connect(self.ref_db_path)
        backendDb.createDb(self.ref_conn)

    def tearDown(self):
        self.conn.close()
        if op.exists(self.ref_db_path):
            os.remove(self.ref_db_path)

    def _vals2str(self, vals):
        '''
        Makes a string from INSERT values.
        Args:
          vals:  a list of tuples, e.g. [(0, 1), (1, 2)].
        Returns:
          a string, e.g. '(0, 1), (1, 2)'.
        '''
        vals_str = []
        for val in vals:
            vals_str.append(','.join(['"%s"' % x for x in val]))
        return ', '.join(['(%s)' % x for x in vals_str])

    def _insertObjectsValue(self, vals, vals_ref):
        ''' 
        Insert values into 'objects' table of the active and ref dbs.
        Args:
          vals, vals_ref:  a tuple with 5 numbes (objectid,x1,y1,width,height).
        '''
        c = self.conn.cursor()
        c_ref = self.ref_conn.cursor()
        s = 'objects(objectid,x1,y1,width,height)'
        if len(vals):
            c.execute('INSERT INTO %s VALUES %s' % (s, self._vals2str(vals)))
        if len(vals_ref):
            c_ref.execute('INSERT INTO %s VALUES %s' %
                          (s, self._vals2str(vals_ref)))
        self.ref_conn.commit()
        self.ref_conn.close()

    def _insertPolygonsValue(self, vals, vals_ref):
        ''' Insert values into 'polygons' table of the active and ref dbs.
        Args:
          vals, vals_ref:  a tuple with 3 numbers (id,x,y).
        '''
        c = self.conn.cursor()
        c_ref = self.ref_conn.cursor()
        s = 'polygons(id,x,y)'
        if len(vals):
            c.execute('INSERT INTO %s VALUES %s' % (s, self._vals2str(vals)))
        if len(vals_ref):
            c_ref.execute('INSERT INTO %s VALUES %s' %
                          (s, self._vals2str(vals_ref)))
        self.ref_conn.commit()
        self.ref_conn.close()

    def test_objects(self):
        '''
        Sync x1, x2, y1, or y2  in 'objects', whichever has changed.
        E.g.:
          - if x2 = x1 + width has changed, update width.
          - if x1 has changed, update both x1 and width.
        '''
        #  Objects with identical coordinates.
        vals = [(0, 10, 20, 30, 40), (1, 10, 20, 30, 40), (2, 10, 20, 30, 40),
                (3, 10, 20, 30, 40), (4, 10, 20, 30, 40), (5, 10, 20, 30, 40),
                (6, 10, 20, 30, 40), (7, 10, 20, 30, 40), (8, 10, 20, 30, 40),
                (9, 10, 20, 30, 40)]
        vals_ref = [
            (0, 10, 20, 30, 40),  # No change.
            (1, 10.1, 20, 29.9, 40),  # Changed a bit only x1 (x2 is fixed).
            (2, 10, 20.2, 30, 39.8),  # Changed a bit only y1 (y2 is fixed).
            (3, 10, 20, 30.3, 40),  # Changed a bit only x2 (via width).
            (4, 10, 20, 30, 40.4),  # Changed a bit only y2 (via height).
            (5, 10.1, 20, 35, 40),  # x1 changed a bit, but x2 changed a lot.
            (6, 10, 20.1, 30, 45),  # y1 changed a bit, but y2 changed a lot.
            (7, 10.1, 20.2, 30, 40),  # x1,y1,x2,y2 changed a bit.
            (8, 15, 25, 35, 45),  # x1,y1,x2,y2 changed a lot.
            # objectid=9 is new in active db (missing from ref_db).
            (10, 10, 20, 30, 40),  # missing from active db.
        ]
        self._insertObjectsValue(vals, vals_ref)
        c = self.conn.cursor()
        dbModify.syncRoundedCoordinatesWithDb(
            c, argparse.Namespace(ref_db_file=self.ref_db_path, epsilon=1.))

        vals_expected = [
            (0, 10, 20, 30, 40),  # No change.
            (1, 10.1, 20, 29.9, 40
             ),  # x1 is taken from ref_db, width is computed to keep x2 fixed.
            (2, 10, 20.2, 30, 39.8
             ),  # y1 is taken from ref_db, height is computed to keep y2 fixed
            (3, 10, 20, 30.3, 40),  # only width is taken from ref_db.
            (4, 10, 20, 30, 40.4),  # only height is taken from ref_db.
            (5, 10.1, 20, 35, 40
             ),  # x1 is taken from ref_db, width is computed to keep x2 fixed.
            (6, 10, 20.1, 30, 45
             ),  # y1 is taken from ref_db, height is computed to keep y2 fixed
            (7, 10.1, 20.2, 30, 40),  # all are taken from ref_db.
            (8, 10, 20, 30, 40),  # all are fixed.
            (9, 10, 20, 30, 40),  # all are taken from active db since it's new
        ]
        c.execute('SELECT objectid,x1,y1,width,height FROM objects')
        entries = c.fetchall()
        self.assertEqual(entries, vals_expected)

    def test_polygons(self):
        ''' Sync x or y in 'polygons', whichever has changed. '''
        vals = [(0, 10, 20), (1, 10, 20), (2, 10, 20), (3, 10, 20),
                (4, 10, 20), (5, 10, 20)]
        vals_ref = [
            (0, 10, 20),  # No change.
            (1, 10.1, 20),  # Changed a bit only x.
            (2, 10, 20.2),  # Changed a bit only y.
            (3, 10.1, 20.2),  # x,y changed a bit.
            (4, 15, 25),  # x,y changed a lot.
            # objectid=5 is new in active db (missing from ref_db).
            (6, 10, 20),  # missing from active db.
        ]
        self._insertPolygonsValue(vals, vals_ref)
        c = self.conn.cursor()
        dbModify.syncRoundedCoordinatesWithDb(
            c, argparse.Namespace(ref_db_file=self.ref_db_path, epsilon=1.))

        vals_expected = [
            (0, 10, 20),  # No change.
            (1, 10.1, 20),  # x is taken from ref_db.
            (2, 10, 20.2),  # y is taken from ref_db.
            (3, 10.1, 20.2),  # all are taken from ref_db.
            (4, 10, 20),  # all are fixed.
            (5, 10, 20),  # all are taken from active db since it's new
        ]
        c.execute('SELECT id,x,y FROM polygons')
        entries = c.fetchall()
        self.assertEqual(entries, vals_expected)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
