import os, os.path as op
import logging
import sqlite3
import shutil
import progressbar
import unittest
import argparse
import pprint
import tempfile
import numpy as np
import cv2
import nose

from shuffler.backend import backend_db
from shuffler.operations.datasets import labelme
from shuffler.operations import modify


class Test_exportLabelme_importLabelme_synthetic(unittest.TestCase):
    '''
    Check that running exportLabelme, importLabelme, and syncRoundedCoordinatesWithDb in
    sequence has no effect.
    '''

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()
        self.rootdir = self.work_dir  # rootdir is defined for readability.
        # Image dir is another layer of directories in order to check option
        # "--dirtree_level_for_name".
        self.image_dir = op.join(self.work_dir, 'my/images')
        os.makedirs(self.image_dir)
        # Image format is png in order to check that it changes at export.
        self.imagefile = 'my/images/image().png'

        self.imwidth, self.imheight = 400, 30
        image = np.zeros((self.imheight, self.imwidth, 3), np.uint8)
        cv2.imwrite(op.join(self.rootdir, self.imagefile), image)
        assert op.exists(op.join(self.rootdir, self.imagefile))

        self.exported_db_file = op.join(self.rootdir, 'exported_db_file.db')
        self.conn = sqlite3.connect(self.exported_db_file)
        backend_db.createDb(self.conn)
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES (?)',
                  (self.imagefile, ))
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES (?,123,40,20,30,10,"myclass")', (self.imagefile, ))
        # Polygons are consistent with the object.
        c.execute('INSERT INTO polygons(objectid,x,y) '
                  'VALUES (123,40,20), (123,70,20), (123,70,30)')
        # Save changes and make a copy to allow the comparison after the import.
        self.conn.commit()
        self.old_db_file = op.join(self.rootdir, 'old_db_file.db')
        shutil.copyfile(self.exported_db_file, self.old_db_file)

    def tearDown(self):
        if op.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

    def test_regular(self):
        #
        # Export.
        #
        c = self.conn.cursor()
        args = argparse.Namespace(rootdir=self.rootdir,
                                  images_dir=op.join(self.work_dir, 'Images'),
                                  annotations_dir=op.join(
                                      self.work_dir, 'Annotations'),
                                  username='',
                                  folder='',
                                  source_image='',
                                  source_annotation='',
                                  dirtree_level_for_name=2,
                                  fix_invalid_image_names=True,
                                  overwrite=False)
        labelme.exportLabelme(c, args)
        # Save changes to allow the use of this file as ref_db_file at export.
        self.conn.commit()

        #
        # Import.
        #
        conn_new = sqlite3.connect(':memory:')
        backend_db.createDb(conn_new)
        c_new = conn_new.cursor()
        args = argparse.Namespace(rootdir=self.rootdir,
                                  images_dir=op.join(self.work_dir, 'Images'),
                                  annotations_dir=op.join(
                                      self.work_dir, 'Annotations'),
                                  replace=False,
                                  ref_db_file=self.exported_db_file)
        labelme.importLabelme(c_new, args)

        # Sync objectids.
        logging.debug('============ syncObjectidsWithDb =============')
        args = argparse.Namespace(ref_db_file=self.old_db_file,
                                  IoU_threshold=0.9)
        modify.syncObjectidsWithDb(c_new, args)

        # Sync polygons. Polygon names are not preserved by Labelme.
        logging.debug('============ syncPolygonIdsWithDb =============')
        args = argparse.Namespace(ref_db_file=self.old_db_file,
                                  epsilon=1.,
                                  ignore_name=True)
        modify.syncPolygonIdsWithDb(c_new, args)

        # Sync to fix rounding.
        logging.debug('============ syncRoundedCoordinatesWithDb ============')
        args = argparse.Namespace(ref_db_file=self.old_db_file, epsilon=1.)
        modify.syncRoundedCoordinatesWithDb(c_new, args)

        # Check that imagefiles matches.
        conn_old = backend_db.connect(self.old_db_file, 'load_to_memory')
        c_old = conn_old.cursor()
        images_query = 'SELECT imagefile FROM images ORDER BY imagefile'
        c_old.execute(images_query)
        imagefiles_old = c_old.fetchall()
        c_new.execute(images_query)
        imagefiles_new = c_new.fetchall()
        self.assertEqual(imagefiles_old, imagefiles_new)

        # Check that objects match.
        objects_query = ('SELECT objectid,imagefile,x1,y1,width,height,name '
                         'FROM objects ORDER BY objectid')
        c_old.execute(objects_query)
        objects_old = c_old.fetchall()
        c_new.execute(objects_query)
        objects_new = c_new.fetchall()
        self.assertEqual(objects_old, objects_new)

        # Check that polygons match.
        OBJECT_ID_WITH_POLYGONS = 123  # Only this object has polygons in cars db.
        polygons_query = 'SELECT id,objectid,x,y FROM polygons WHERE objectid=? ORDER BY id'
        c_old.execute(polygons_query, (OBJECT_ID_WITH_POLYGONS, ))
        polygons_old = c_old.fetchall()
        logging.debug('old polygons:\n%s', pprint.pformat(polygons_old))
        assert len(polygons_old) == 3, 'SetUp wrote 3 polygons.'
        c_new.execute(polygons_query, (OBJECT_ID_WITH_POLYGONS, ))
        polygons_new = c_new.fetchall()
        logging.debug('new polygons:\n%s', pprint.pformat(polygons_new))
        self.assertEqual(len(polygons_new), 3,
                         'importLabelme mist read the original polygons.')
        self.assertEqual(polygons_old, polygons_new)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
