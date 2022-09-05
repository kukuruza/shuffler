import os, os.path as op
import sqlite3
import progressbar
import unittest
import argparse
import tempfile
import numpy as np
import cv2
import nose

from lib.backend import backendDb
from lib.subcommands.datasets import dbLabelme
from lib.subcommands import dbInfo


class Test_exportLabelme_importLabelme_synthetic(unittest.TestCase):
    def setUp(self):
        ''' Create a database with one iamge and one object. '''

        self.work_dir = tempfile.mkdtemp()
        self.rootdir = self.work_dir  # rootdir is defined for readability.
        # Image dir is another layer of directories in order to check option
        # "--full_imagefile_as_name".
        self.image_dir = op.join(self.work_dir, 'images')
        os.makedirs(self.image_dir)
        # Image format is png in order to check that it changes at export.
        self.imagefile = 'images/image().png'

        self.imwidth, self.imheight = 400, 30
        image = np.zeros((self.imheight, self.imwidth, 3), np.uint8)
        cv2.imwrite(op.join(self.rootdir, self.imagefile), image)
        assert op.exists(op.join(self.rootdir, self.imagefile))

        self.ref_db_file = op.join(self.rootdir, 'ref_db_file.db')
        self.conn = sqlite3.connect(self.ref_db_file)
        backendDb.createDb(self.conn)
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES (?)',
                  (self.imagefile, ))
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES (?,123,40,20,30,10,"myclass")', (self.imagefile, ))
        # Polygons are consistent with the object.
        c.execute('INSERT INTO polygons(objectid,x,y) '
                  'VALUES (123,40,20), (123,70,20), (123,70,30)')

    def test_noPolygons(self):
        #
        # Export.
        #
        c_old = self.conn.cursor()
        args = argparse.Namespace(rootdir=self.rootdir,
                                  images_dir=op.join(self.work_dir, 'Images'),
                                  annotations_dir=op.join(
                                      self.work_dir, 'Annotations'),
                                  username='',
                                  folder='',
                                  source_image='',
                                  source_annotation='',
                                  full_imagefile_as_name=True,
                                  fix_invalid_image_names=True,
                                  overwrite=False)
        dbLabelme.exportLabelme(c_old, args)
        # Save changes to allow the use of this file as ref_db_file at export.
        self.conn.commit()

        # Check that the exported image name is correct
        c_old.execute('SELECT imagefile FROM images')
        imagefile_old = c_old.fetchone()
        # The export must make the following changes:
        #   1. Dir "images" is a part of the file name.
        #   2. Invalid characters "(" and ")" are replaces with "_".
        #   3. png is replaces with jpg.
        self.assertEqual(imagefile_old, ('Images/images_image__.jpg', ))
        # Make sure exported files exist.
        self.assertTrue(
            op.exists(op.join(self.work_dir, 'Images/images_image__.jpg')))
        self.assertTrue(
            op.exists(op.join(self.work_dir,
                              'Annotations/images_image__.xml')))

        #
        # Import.
        #
        conn_new = sqlite3.connect(':memory:')
        backendDb.createDb(conn_new)
        c_new = conn_new.cursor()
        args = argparse.Namespace(rootdir=self.rootdir,
                                  images_dir=op.join(self.work_dir, 'Images'),
                                  annotations_dir=op.join(
                                      self.work_dir, 'Annotations'),
                                  replace=False,
                                  ref_db_file=self.ref_db_file)
        dbLabelme.importLabelme(c_new, args)

        # Check that imagefiles matches.
        c_new.execute('SELECT imagefile FROM images')
        imagefile_new = c_new.fetchone()
        self.assertEqual(imagefile_new, (self.imagefile, ))

        # Check that objects match.
        c_old.execute('SELECT * FROM objects')
        objects_old = c_old.fetchall()
        c_new.execute('SELECT * FROM objects')
        objects_new = c_new.fetchall()

        args = argparse.Namespace(tables=['objects'], limit=100)
        dbInfo.dumpDb(c_old, args)
        dbInfo.dumpDb(c_new, args)

        self.assertEqual(len(objects_old), 1,
                         'exportLabelme must keep the original object.')
        self.assertEqual(len(objects_new), 1,
                         'importLabelme mist read the original object.')
        bbox_old = backendDb.objectField(objects_old[0], 'bbox')
        bbox_new = backendDb.objectField(objects_new[0], 'bbox')
        self.assertListEqual(bbox_old, bbox_new)
        name_old = backendDb.objectField(objects_old[0], 'name')
        name_new = backendDb.objectField(objects_new[0], 'name')
        self.assertEqual(name_old, name_new)

        # Check that objects match.
        c_old.execute('SELECT x,y FROM polygons')
        polygons_old = c_old.fetchall()
        c_new.execute('SELECT x,y FROM polygons')
        polygons_new = c_new.fetchall()

        self.assertEqual(len(polygons_old), 3,
                         'exportLabelme must keep the original polygons.')
        self.assertEqual(len(polygons_new), 3,
                         'importLabelme mist read the original polygons.')
        self.assertTupleEqual(polygons_old[0], polygons_new[0])


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
