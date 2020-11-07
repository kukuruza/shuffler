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


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
