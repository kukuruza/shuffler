import pytest
import os, os.path as op
import logging
import sqlite3
import shutil
import argparse
import pprint
import tempfile
import numpy as np
import cv2

from shuffler.backend import backend_db
from shuffler.operations.datasets import labelme
from shuffler.operations import modify


class Test_ExportLabelme_ImportLabelme_SyntheticDb:
    '''
    Check that running exportLabelme, importLabelme, and syncRoundedCoordinatesWithDb in
    sequence has no effect.
    '''
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        yield work_dir
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    def test_regular(self, work_dir):
        #
        # Prepare.
        #
        rootdir = work_dir  # rootdir is defined for readability.
        # Image dir is another layer of directories in order to check option
        # "--dirtree_level_for_name".
        image_dir = op.join(work_dir, 'my/images')
        os.makedirs(image_dir)
        # Image format is png in order to check that it changes at export.
        imagefile = 'my/images/image().png'

        imwidth, imheight = 400, 30
        image = np.zeros((imheight, imwidth, 3), np.uint8)
        cv2.imwrite(op.join(rootdir, imagefile), image)
        assert op.exists(op.join(rootdir, imagefile))

        exported_db_file = op.join(rootdir, 'exported_db_file.db')
        conn = sqlite3.connect(exported_db_file)
        backend_db.createDb(conn)
        c = conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES (?)', (imagefile, ))
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES (?,123,40,20,30,10,"myclass")', (imagefile, ))
        # Polygons are consistent with the object.
        c.execute('INSERT INTO polygons(objectid,x,y) '
                  'VALUES (123,40,20), (123,70,20), (123,70,30)')
        # Save changes and make a copy to allow the comparison after the import.
        conn.commit()
        old_db_file = op.join(rootdir, 'old_db_file.db')
        shutil.copyfile(exported_db_file, old_db_file)

        #
        # Export.
        #
        c = conn.cursor()
        args = argparse.Namespace(rootdir=rootdir,
                                  images_dir=op.join(work_dir, 'Images'),
                                  annotations_dir=op.join(
                                      work_dir, 'Annotations'),
                                  username='',
                                  folder='',
                                  source_image='',
                                  source_annotation='',
                                  dirtree_level_for_name=2,
                                  fix_invalid_image_names=True,
                                  overwrite=False)
        labelme.exportLabelme(c, args)
        # Save changes to allow the use of this file as ref_db_file at export.
        conn.commit()

        #
        # Import.
        #
        conn_new = sqlite3.connect(':memory:')
        backend_db.createDb(conn_new)
        c_new = conn_new.cursor()
        args = argparse.Namespace(rootdir=rootdir,
                                  images_dir=op.join(work_dir, 'Images'),
                                  annotations_dir=op.join(
                                      work_dir, 'Annotations'),
                                  replace=False,
                                  ref_db_file=exported_db_file)
        labelme.importLabelme(c_new, args)

        # Sync objectids.
        logging.debug('============ syncObjectidsWithDb =============')
        args = argparse.Namespace(ref_db_file=old_db_file, IoU_threshold=0.9)
        modify.syncObjectidsWithDb(c_new, args)

        # Sync polygons. Polygon names are not preserved by Labelme.
        logging.debug('============ syncPolygonIdsWithDb =============')
        args = argparse.Namespace(ref_db_file=old_db_file,
                                  epsilon=1.,
                                  ignore_name=True)
        modify.syncPolygonIdsWithDb(c_new, args)

        # Sync to fix rounding.
        logging.debug('============ syncRoundedCoordinatesWithDb ============')
        args = argparse.Namespace(ref_db_file=old_db_file, epsilon=1.)
        modify.syncRoundedCoordinatesWithDb(c_new, args)

        # Check that imagefiles matches.
        conn_old = backend_db.connect(old_db_file, 'load_to_memory')
        c_old = conn_old.cursor()
        images_query = 'SELECT imagefile FROM images ORDER BY imagefile'
        c_old.execute(images_query)
        imagefiles_old = c_old.fetchall()
        c_new.execute(images_query)
        imagefiles_new = c_new.fetchall()
        assert imagefiles_old == imagefiles_new

        # Check that objects match.
        objects_query = ('SELECT objectid,imagefile,x1,y1,width,height,name '
                         'FROM objects ORDER BY objectid')
        c_old.execute(objects_query)
        objects_old = c_old.fetchall()
        c_new.execute(objects_query)
        objects_new = c_new.fetchall()
        assert objects_old == objects_new

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
        assert len(polygons_new
                   ) == 3, 'importLabelme must read the original polygons.'
        assert polygons_old == polygons_new
