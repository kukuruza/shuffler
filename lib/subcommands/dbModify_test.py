import os, os.path as op
import logging
import sqlite3
import shutil
import progressbar
import unittest
import argparse
import pprint
import tempfile
import mock
import numpy as np
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
                  'VALUES ("a",0,10,"cat",0.1), ("b",1,20,"dog",0.2)')
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
                  'VALUES (0,"newval","2")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='objectid',
                                  properties_key='newval')
        dbModify.propertyToObjectsField(c, args)

        # Verify the "objects" table.
        c.execute('SELECT imagefile,objectid FROM objects')
        expected = [("a", 2), ("b", 1)]
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
                    (2, "newval", "2")]
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
        expected = [(0, "sheep"), (1, "dog")]
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
        expected = [(0, 50), (1, 20)]
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
        expected = [(0, 0.5), (1, 0.2)]
        self.assertEqual(set(c.fetchall()), set(expected))


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
