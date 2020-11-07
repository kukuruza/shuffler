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
from lib.subcommands import dbFilter
from lib.utils import testUtils


class Test_filterObjectsInsideCertainObjects_SyntheticDb(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)

    def test_empty(self):
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        args = argparse.Namespace(where_shadowing_objects='TRUE',
                                  where_objects='TRUE')
        # Should succeed without issues.
        dbFilter.filterObjectsInsideCertainObjects(c, args)

    def test_1shadowing_0others(self):
        '''
        A single shadowing object, no other objects.
        Nothing should be deleted.
        '''
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image0",1,40,20,40,20,"shadowing")')
        args = argparse.Namespace(where_shadowing_objects='TRUE',
                                  where_objects='TRUE')
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        c.execute('SELECT objectid FROM objects')
        object_ids = c.fetchall()
        self.assertEqual(object_ids, [(1, )])

    def test_1shadowingBox_1outsideBox(self):
        '''
        1 shadowing object (via a box), 1 other objects (via a box.)
        The other object is OUTSIDE the shadowing one. It should NOT be deleted.
        '''
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image0",1,40,20,40,20,"shadowing")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("image0",2,140,120,40,20)')
        args = argparse.Namespace(where_shadowing_objects='name="shadowing"',
                                  where_objects='TRUE')
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        c.execute('SELECT objectid FROM objects')
        object_ids = c.fetchall()
        self.assertEqual(object_ids, [(1, ), (2, )])

    def test_1shadowingBox_1insideBox(self):
        '''
        1 shadowing object (via a box), 1 other objects (via a box.)
        The other object is INSIDE the shadowing one. It should be deleted.
        '''
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image0",1,40,20,40,20,"shadowing")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("image0",2,40,20,40,20)')
        args = argparse.Namespace(where_shadowing_objects='name="shadowing"',
                                  where_objects='TRUE')
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        c.execute('SELECT objectid FROM objects')
        object_ids = c.fetchall()
        self.assertEqual(object_ids, [(1, )])

    def test_1shadowingBox_1goodBox(self):
        '''
        1 shadowing object (via a box), 1 other GOOD objects (via a box.)
        The other object is INSIDE the shadowing one.
        It should NOT be deleted because it is good.
        '''
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image0",1,40,20,40,20,"shadowing")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image0",2,40,20,40,20,"keep")')
        args = argparse.Namespace(where_shadowing_objects='name="shadowing"',
                                  where_objects='name!="keep"')
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        c.execute('SELECT objectid FROM objects')
        object_ids = c.fetchall()
        self.assertEqual(object_ids, [(1, ), (2, )])

    def test_1shadowingBox_1insidePolygon(self):
        '''
        1 shadowing object (via a box), 1 other objects (via a polygon.)
        The other object is INSIDE the shadowing one. It should be deleted.
        '''
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image0",1,40,20,40,20,"shadowing")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid) VALUES ("image0",2)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (2,40,20)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (2,80,20)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (2,80,40)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (2,40,40)')
        args = argparse.Namespace(where_shadowing_objects='name="shadowing"',
                                  where_objects='TRUE')
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        c.execute('SELECT objectid FROM objects')
        object_ids = c.fetchall()
        self.assertEqual(object_ids, [(1, )])

    def test_1shadowingBox_1outsidePolygon(self):
        '''
        1 shadowing object (via a box), 1 other objects (via a polygon.)
        The other object is OUTSIDE the shadowing one. It should NOT be deleted.
        '''
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image0",1,40,20,40,20,"shadowing")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid) VALUES ("image0",2)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (2,140,120)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (2,180,120)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (2,180,140)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (2,140,140)')
        args = argparse.Namespace(where_shadowing_objects='name="shadowing"',
                                  where_objects='TRUE')
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        c.execute('SELECT objectid FROM objects')
        object_ids = c.fetchall()
        self.assertEqual(object_ids, [(1, ), (2, )])

    def test_1shadowingPolygon_1insideBox(self):
        '''
        1 shadowing object (via a polygon), 1 other objects (via a box.)
        The other object is INSIDE the shadowing one. It should be deleted.
        '''
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute('INSERT INTO objects(imagefile,objectid,name) '
                  'VALUES ("image0",1,"shadowing")')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (1,40,20)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (1,80,20)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (1,80,40)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (1,40,40)')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("image0",2,40,20,40,20)')
        args = argparse.Namespace(where_shadowing_objects='name="shadowing"',
                                  where_objects='TRUE')
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        c.execute('SELECT objectid FROM objects')
        object_ids = c.fetchall()
        self.assertEqual(object_ids, [(1, )])

    def test_1shadowingPolygon_1outsideBox(self):
        '''
        1 shadowing object (via a polygon), 1 other objects (via a box.)
        The other object is OUTSIDE the shadowing one. It NOT should be deleted.
        '''
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute('INSERT INTO objects(imagefile,objectid,name) '
                  'VALUES ("image0",1,"shadowing")')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (1,140,20)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (1,180,20)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (1,180,40)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (1,140,40)')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("image0",2,40,20,40,20)')
        args = argparse.Namespace(where_shadowing_objects='name="shadowing"',
                                  where_objects='TRUE')
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        c.execute('SELECT objectid FROM objects')
        object_ids = c.fetchall()
        self.assertEqual(object_ids, [(1, ), (2, )])


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
