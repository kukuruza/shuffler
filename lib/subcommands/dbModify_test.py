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


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
