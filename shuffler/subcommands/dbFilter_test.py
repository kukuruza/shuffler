import os, os.path as op
import sqlite3
import unittest
import argparse
import tempfile
import progressbar
import nose

from shuffler.backend import backendDb
from shuffler.subcommands import dbFilter
from shuffler.utils import testUtils


class Test_filterObjectsInsideCertainObjects_SyntheticDb(unittest.TestCase):

    # TODO: Add tests with invert=True.

    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)

    def test_empty(self):
        ''' Should succeed without issues. '''
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        args = argparse.Namespace(where_shadowing_objects='TRUE',
                                  where_objects='TRUE',
                                  invert=False)
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        args = argparse.Namespace(where_shadowing_objects='TRUE',
                                  where_objects='TRUE',
                                  invert=True)
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
                                  where_objects='TRUE',
                                  invert=False)
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
                                  where_objects='TRUE',
                                  invert=False)
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
                                  where_objects='TRUE',
                                  invert=False)
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
                                  where_objects='name!="keep"',
                                  invert=False)
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
                                  where_objects='TRUE',
                                  invert=False)
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
                                  where_objects='TRUE',
                                  invert=False)
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
                                  where_objects='TRUE',
                                  invert=False)
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
                                  where_objects='TRUE',
                                  invert=False)
        dbFilter.filterObjectsInsideCertainObjects(c, args)
        c.execute('SELECT objectid FROM objects')
        object_ids = c.fetchall()
        self.assertEqual(object_ids, [(1, ), (2, )])


class Test_filterBadImages_CarsDb(testUtils.Test_carsDb):
    def setUp(self):
        testUtils.Test_carsDb.setUp(self)
        # Make a corrupted image file.
        self.bad_jpg_path = tempfile.NamedTemporaryFile(suffix='.jpg').name
        with open(self.bad_jpg_path, 'w') as f:
            f.write('I am a corrupted image file.')

    def tearDown(self):
        if op.exists(self.bad_jpg_path):
            os.remove(self.bad_jpg_path)
        testUtils.Test_carsDb.tearDown(self)

    def test_single_thread_all_ok(self):
        ''' Tests the single-thread mode when all images are ok. '''

        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            force_single_thread=True)
        dbFilter.filterBadImages(c, args)
        c.execute('SELECT COUNT(1) FROM images')
        self.assertEqual(c.fetchone()[0], 3)

    def test_parallel_all_ok(self):
        ''' Tests the parallel mode when all images are ok. '''

        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            force_single_thread=False)
        dbFilter.filterBadImages(c, args)
        c.execute('SELECT COUNT(1) FROM images')
        self.assertEqual(c.fetchone()[0], 3)

    def test_single_thread_missing(self):
        ''' Tests deleting a missing image in the single-thread mode. '''

        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("non-existent.jpg")')
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            force_single_thread=True)
        dbFilter.filterBadImages(c, args)
        c.execute('SELECT COUNT(1) FROM images')
        self.assertEqual(c.fetchone()[0], 3)

    def test_single_thread_corrupted(self):
        ''' Tests deleting a corrupted image in the single-thread mode. '''

        # Add the corrrupted image to the db.
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES (?)', (op.relpath(
            self.bad_jpg_path, testUtils.Test_carsDb.CARS_DB_ROOTDIR), ))

        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            force_single_thread=True)
        dbFilter.filterBadImages(c, args)
        c.execute('SELECT COUNT(1) FROM images')
        self.assertEqual(c.fetchone()[0], 3)

    def test_parallel_corrupted(self):
        ''' Tests deleting a corrupted image in the parallel mode. '''

        # Add the corrrupted image to the db.
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES (?)', (op.relpath(
            self.bad_jpg_path, testUtils.Test_carsDb.CARS_DB_ROOTDIR), ))

        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            force_single_thread=False)
        dbFilter.filterBadImages(c, args)
        c.execute('SELECT COUNT(1) FROM images')
        self.assertEqual(c.fetchone()[0], 3)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
