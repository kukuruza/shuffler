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

from lib.backend import backendDb
from lib.subcommands import dbStamps


class Test_emptyDb(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)

    def tearDown(self):
        self.conn.close()


class Test_extractNumberIntoProperty(Test_emptyDb):
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


if __name__ == '__main__':
    import nose
    nose.runmodule()
