import os, sys, os.path as op
import shutil
import random
import logging
import numpy as np
import unittest
import tempfile

from lib.utils import util


class TestCopyWithBackup(unittest.TestCase):
    def setUp(self):
        self.work_dir = tempfile.mkdtemp()
        with open(op.join(self.work_dir, 'from.txt'), 'w') as f:
            f.write('from')

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_failsIfFileNotExists(self):
        from_path = op.join(self.work_dir, 'not_exists.txt')
        to_path = op.join(self.work_dir, 'to.txt')
        with self.assertRaises(FileNotFoundError):
            util.copyWithBackup(from_path, to_path)

    def test_copiesWithoutNeedForBackup(self):
        from_path = op.join(self.work_dir, 'from.txt')
        to_path = op.join(self.work_dir, 'to.txt')
        util.copyWithBackup(from_path, to_path)
        self.assertTrue(op.exists(from_path))
        self.assertTrue(op.exists(to_path))

    def test_copiesWithBackup(self):
        from_path = op.join(self.work_dir, 'from.txt')
        to_path = op.join(self.work_dir, 'to.txt')
        backup_path = op.join(self.work_dir, 'to.backup.txt')
        # Make an existing file.
        with open(to_path, 'w') as f:
            f.write('old')
        util.copyWithBackup(from_path, to_path)
        # Make sure the contents are overwritten.
        self.assertTrue(op.exists(to_path))
        with open(to_path) as f:
            s = f.readline()
            self.assertEqual(s, 'from')
        # Make sure the old file was backed up.
        self.assertTrue(op.exists(backup_path))
        with open(backup_path) as f:
            s = f.readline()
            self.assertEqual(s, 'old')

    def test_copiestoItself(self):
        from_path = op.join(self.work_dir, 'from.txt')
        backup_path = op.join(self.work_dir, 'from.backup.txt')
        util.copyWithBackup(from_path, from_path)
        # Make sure the contents are the same.
        self.assertTrue(op.exists(from_path))
        with open(from_path) as f:
            s = f.readline()
            self.assertEqual(s, 'from')
        # Make sure the old file was backed up.
        self.assertTrue(op.exists(backup_path))
        with open(backup_path) as f:
            s = f.readline()
            self.assertEqual(s, 'from')


class TestBbox2roi(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(utilBoxes.bbox2roi([1, 2, 3, 4]), [2, 1, 6, 4])
        self.assertEqual(utilBoxes.bbox2roi((1, 2, 3, 4)), [2, 1, 6, 4])

    def test_zeroDims(self):
        self.assertEqual(utilBoxes.bbox2roi([1, 2, 0, 0]), [2, 1, 2, 1])

    def test_notSequence(self):
        with self.assertRaises(TypeError):
            utilBoxes.bbox2roi(42)

    def test_LessThanFourNumbers(self):
        with self.assertRaises(ValueError):
            utilBoxes.bbox2roi([42])

    def test_MoreThanFourNumbers(self):
        with self.assertRaises(ValueError):
            utilBoxes.bbox2roi([42, 42, 42, 42, 42])

    def test_NotNumbers(self):
        with self.assertRaises(TypeError):
            utilBoxes.bbox2roi(['a', 'b', 'c', 'd'])

    def test_negativeDims(self):
        with self.assertRaises(ValueError):
            utilBoxes.bbox2roi([1, 2, 3, -1])


class TestRoi2Bbox(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(utilBoxes.roi2bbox([2, 1, 6, 4]), [1, 2, 3, 4])
        self.assertEqual(utilBoxes.roi2bbox((2, 1, 6, 4)), [1, 2, 3, 4])

    def test_zeroDims(self):
        self.assertEqual(utilBoxes.roi2bbox([2, 1, 2, 1]), [1, 2, 0, 0])

    def test_notSequence(self):
        with self.assertRaises(TypeError):
            utilBoxes.roi2bbox(42)

    def test_LessThanFourNumbers(self):
        with self.assertRaises(ValueError):
            utilBoxes.roi2bbox([42])

    def test_MoreThanFourNumbers(self):
        with self.assertRaises(ValueError):
            utilBoxes.roi2bbox([42, 42, 42, 42, 42])

    def test_NotNumbers(self):
        with self.assertRaises(TypeError):
            utilBoxes.roi2bbox(['a', 'b', 'c', 'd'])

    def test_negativeDims(self):
        with self.assertRaises(ValueError):
            utilBoxes.roi2bbox([2, 1, 1, 2])


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    unittest.main()
