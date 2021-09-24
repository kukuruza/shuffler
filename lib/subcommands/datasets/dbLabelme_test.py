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
from lib.subcommands.datasets import dbLabelme
from lib.subcommands import dbInfo
from lib.utils import testUtils


class Test_pipeline_carsDb(testUtils.Test_carsDb):
    def setUp(self):
        super(Test_pipeline_carsDb, self).setUp()
        self.temp_dir = op.join(tempfile.gettempdir(), 'dbLabelme_test')
        assert not op.exists(self.temp_dir)
        os.mkdir(self.temp_dir)

    def tearDown(self):
        super(Test_pipeline_carsDb, self).tearDown()
        if op.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_consistency(self):
        c_old = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            images_dir=op.join(self.temp_dir, 'Images'),
            annotations_dir=op.join(self.temp_dir, 'Annotations'),
            username='dummy_username',
            folder='dummy_labelme_folder',
            source_image='dummy_source_image',
            source_annotation='dummy_source_annotation',
            fix_invalid_image_names=False,
            overwrite=False)
        dbLabelme.exportLabelme(c_old, args)

        conn_new = sqlite3.connect(':memory:')
        backendDb.createDb(conn_new)
        c_new = conn_new.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            images_dir=op.join(self.temp_dir, 'Images'),
            annotations_dir=op.join(self.temp_dir, 'Annotations'),
            replace=False,
            with_display=False)
        dbLabelme.importLabelme(c_new, args)

        # Check that imagefiles matches.
        c_old.execute('SELECT imagefile FROM images')
        imagefiles_old = c_old.fetchall()
        c_new.execute('SELECT imagefile FROM images')
        imagefiles_new = c_new.fetchall()
        self.assertEqual(set(imagefiles_old), set(imagefiles_new))

        # Check that objects match.
        c_old.execute('SELECT * FROM objects WHERE imagefile LIKE "%000000%"')
        objects_old = c_old.fetchall()
        c_new.execute('SELECT * FROM objects WHERE imagefile LIKE "%000000%"')
        objects_new = c_new.fetchall()

        args = argparse.Namespace(tables=['objects'], limit=100)
        dbInfo.dumpDb(c_old, args)
        dbInfo.dumpDb(c_new, args)

        self.assertEqual(len(objects_old), len(objects_new))
        bbox_old = backendDb.objectField(objects_old[0], 'bbox')
        bbox_new = backendDb.objectField(objects_new[0], 'bbox')
        self.assertListEqual(bbox_old, bbox_new)
        name_old = backendDb.objectField(objects_old[0], 'name')
        name_new = backendDb.objectField(objects_new[0], 'name')
        self.assertEqual(name_old, name_new)

    # TODO: def test_exportToLabelme(). Test that:
    #   - Recorded images are the same as source images.
    #   - Check their extensions.
    #   - Check fixed files names.


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
