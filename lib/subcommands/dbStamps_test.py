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
import simplejson as json

from lib.backend import backendDb
from lib.subcommands import dbStamps
from lib.utils import testUtils


class Test_extractNumberIntoProperty(testUtils.Test_emptyDb):
    def test_one_number_middle(self):
        c = self.conn.cursor()
        # Repeated twice.
        c.execute("INSERT INTO objects(name) VALUES ('my123number')")
        c.execute("INSERT INTO objects(name) VALUES ('my123number')")
        args = argparse.Namespace(property='number')
        dbStamps.extractNumberIntoProperty(c, args)
        # Check names.
        c.execute("SELECT name FROM objects")
        names = c.fetchall()
        self.assertEqual(len(names), 2)  # repeated twice
        for name, in names:
            self.assertEqual(name, "mynumber")
        # Check properties.
        c.execute("SELECT key,value FROM properties")
        entries = c.fetchall()
        self.assertEqual(len(entries), 2)  # repeated twice
        for key, value in entries:
            self.assertEqual(key, "number")
            self.assertEqual(value, "123")

    def test_one_on_end(self):
        c = self.conn.cursor()
        c.execute("INSERT INTO objects(name) VALUES ('mynumber123')")
        args = argparse.Namespace(property='number')
        dbStamps.extractNumberIntoProperty(c, args)
        # Check names.
        c.execute("SELECT name FROM objects")
        names = c.fetchall()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0][0], "mynumber")
        # Check properties.
        c.execute("SELECT key,value FROM properties")
        entries = c.fetchall()
        self.assertEqual(len(entries), 1)
        for key, value in entries:
            self.assertEqual(key, "number")
            self.assertEqual(value, "123")

    def test_one_on_beginning(self):
        c = self.conn.cursor()
        c.execute("INSERT INTO objects(name) VALUES ('123mynumber')")
        args = argparse.Namespace(property='number')
        dbStamps.extractNumberIntoProperty(c, args)
        # Check names.
        c.execute("SELECT name FROM objects")
        names = c.fetchall()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0][0], "mynumber")
        # Check properties.
        c.execute("SELECT key,value FROM properties")
        entries = c.fetchall()
        self.assertEqual(len(entries), 1)
        for key, value in entries:
            self.assertEqual(key, "number")
            self.assertEqual(value, "123")

    def test_two_numbers(self):
        c = self.conn.cursor()
        c.execute("INSERT INTO objects(name) VALUES ('my123num456ber')")
        args = argparse.Namespace(property='number')
        dbStamps.extractNumberIntoProperty(c, args)
        # Check names.
        c.execute("SELECT name FROM objects")
        names = c.fetchall()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0][0], "mynumber")
        # Check properties.
        c.execute("SELECT key,value FROM properties")
        entries = c.fetchall()
        keys = [e[0] for e in entries]
        values = [e[1] for e in entries]
        self.assertEqual(len(entries), 2)
        self.assertEqual(keys[0], "number")
        self.assertTrue(values[0] in ["123", "456"])
        self.assertEqual(keys[1], "number")
        self.assertTrue(values[1] in ["123", "456"])
        self.assertNotEqual(values[0], values[1])


class Test_syncImagesWithDb_synthetic(testUtils.Test_emptyDb):
    def setUp(self):
        # Make tested db.
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)
        # Make ref db (should be on disk.)
        self.ref_db_path = tempfile.NamedTemporaryFile().name
        self.conn_ref = sqlite3.connect(self.ref_db_path, timeout=10000.0)
        backendDb.createDb(self.conn_ref)
        c_ref = self.conn_ref.cursor()
        self.imagefiles_ref = ['ref_image0', 'ref_image1']
        c_ref.execute("INSERT INTO images(imagefile) VALUES ('ref_image0')")
        c_ref.execute("INSERT INTO images(imagefile) VALUES ('ref_image1')")
        c_ref.execute("INSERT INTO objects(objectid,imagefile,name) "
                      "VALUES (0,'ref_image0','new')")
        c_ref.execute("INSERT INTO objects(objectid,imagefile,name) "
                      "VALUES (1,'ref_image1','new')")
        self.conn_ref.commit()
        self.conn_ref.close()

    def tearDown(self):
        if op.exists(self.ref_db_path):
            os.remove(self.ref_db_path)

    def test_general(self):
        c = self.conn.cursor()
        # Insert an object and an image that is there in the ref_db_file.
        c.execute("INSERT INTO images(imagefile) VALUES ('image0')")
        c.execute("INSERT INTO objects(objectid,imagefile,name) "
                  "VALUES (0,'image0','new')")
        # Run the function.
        args = argparse.Namespace(ref_db_file=self.ref_db_path)
        dbStamps.syncImagesWithDb(c, args)
        # Test images.
        c.execute("SELECT imagefile FROM images")
        imagefiles = [entry[0] for entry in c.fetchall()]
        self.assertEqual(imagefiles, self.imagefiles_ref)
        # Test objects.
        c.execute("SELECT objectid,name FROM objects")
        object_entries = c.fetchall()
        self.assertEqual(object_entries, [(0, 'new')])

    def test_objectNotFound(self):
        c = self.conn.cursor()
        c.execute("INSERT INTO images(imagefile) VALUES ('image0')")
        c.execute("INSERT INTO objects(objectid,imagefile,name) "
                  "VALUES (2,'image0','new')")
        # objectid=2 is not in ref_db_file.
        args = argparse.Namespace(ref_db_file=self.ref_db_path)
        with self.assertRaises(ValueError):
            dbStamps.syncImagesWithDb(c, args)


class Test_getTop1Name(testUtils.Test_emptyDb):
    def test_general(self):
        c = self.conn.cursor()
        c.execute("INSERT INTO objects(objectid,imagefile,name) VALUES "
                  "(0, 'image0', 'okay'), "
                  "(1, 'image0', 'wood / stone'), "
                  "(2, 'image0', 'cat / dog / sheep')")
        # Run the function.
        dbStamps.getTop1Name(c, argparse.Namespace())
        c.execute("SELECT name FROM objects")
        names = [name[0] for name in c.fetchall()]
        self.assertEqual(names, ['okay', 'wood', 'cat'])


class Test_setNumStampOccurancies(testUtils.Test_emptyDb):
    def test_insert_and_update(self):
        c = self.conn.cursor()

        # The 1st time (check insert).
        c.execute("INSERT INTO objects(name) VALUES ('cat'), ('dog'), ('cat')")
        dbStamps.setNumStampOccurancies(c, argparse.Namespace())
        c.execute('SELECT o.name,value FROM objects o INNER JOIN properties p '
                  ' ON o.objectid == p.objectid WHERE key="num_instances"')
        values = set(c.fetchall())
        # 2 for cats and 1 for dog.
        self.assertEqual(values, set([('cat', '2')] * 2 + [('dog', '1')] * 1))

        # The 2nd time (check update).
        c.execute('INSERT INTO objects(name) VALUES ("cat"), ("dog"), ("cat")')
        dbStamps.setNumStampOccurancies(c, argparse.Namespace())
        c.execute('SELECT o.name,value FROM objects o INNER JOIN properties p '
                  ' ON o.objectid == p.objectid WHERE key="num_instances"')
        values = set(c.fetchall())
        # 4 for cats and 2 for dog.
        self.assertEqual(values, set([('cat', '4')] * 4 + [('dog', '2')] * 2))


class Test_encodeNames(testUtils.Test_carsDb):
    def setUp(self):
        super().setUp()
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir)
        super().tearDown()

    def test_general(self):
        c = self.conn.cursor()

        # Add the special case.
        c.execute("INSERT INTO objects(name) VALUES ('??+'), ('??+'), ('??+')")
        # Add the non-used pages.
        c.execute("INSERT INTO objects(name) VALUES ('page'), ('page_r')")

        encoding_json_file = op.join(self.work_dir, 'encoding.json')
        dbStamps.encodeNames(
            c, argparse.Namespace(encoding_json_file=encoding_json_file))

        c.execute('SELECT o.name,value FROM objects o INNER JOIN properties p '
                  ' ON o.objectid == p.objectid WHERE key="name_id"')
        entries = set(c.fetchall())

        self.assertEqual(entries,
                         set([('bus', '1'), ('car', '0'), ('??+', '-1')]))

        self.assertTrue(op.exists(encoding_json_file))
        encoding = json.load(open(encoding_json_file))
        self.assertTrue('bus' in encoding)
        self.assertEqual(len(encoding), 3, encoding)
        self.assertEqual(encoding['car'], 0, encoding)
        self.assertEqual(encoding['bus'], 1, encoding)
        self.assertEqual(encoding['??+'], -1, encoding)


if __name__ == '__main__':
    import nose
    nose.runmodule()
