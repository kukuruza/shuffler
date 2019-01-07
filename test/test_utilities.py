import os, sys, os.path as op
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import shutil
import random
import logging
import numpy as np
import unittest

from lib import util


class TestCopyWithBackup (unittest.TestCase):

  if os.name == 'nt':
    WORK_DIR = op.join(op.dirname(op.realpath(__file__)), 'tmp/TestCopyWithBackup')
  else:
    WORK_DIR = '/tmp/TestCopyWithBackup'

  def setUp (self):
    if not op.exists(self.WORK_DIR):
      os.makedirs(self.WORK_DIR)
    with open(op.join(self.WORK_DIR, 'from.txt'), 'w') as f:
      f.write('from')

  def tearDown(self):
    shutil.rmtree(self.WORK_DIR)

  def test_failsIfFileNotExists(self):
    from_path = op.join(self.WORK_DIR, 'not_exists.txt')
    to_path = op.join(self.WORK_DIR, 'to.txt')
    with self.assertRaises(FileNotFoundError):
      util.copyWithBackup(from_path, to_path)

  def test_copiesWithoutNeedForBackup(self):
    from_path = op.join(self.WORK_DIR, 'from.txt')
    to_path = op.join(self.WORK_DIR, 'to.txt')
    util.copyWithBackup(from_path, to_path)
    self.assertTrue(op.exists(from_path))
    self.assertTrue(op.exists(to_path))

  def test_copiesWithBackup(self):
    from_path = op.join(self.WORK_DIR, 'from.txt')
    to_path = op.join(self.WORK_DIR, 'to.txt')
    backup_path = op.join(self.WORK_DIR, 'to.backup.txt')
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
    from_path = op.join(self.WORK_DIR, 'from.txt')
    backup_path = op.join(self.WORK_DIR, 'from.backup.txt')
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


class TestBbox2roi (unittest.TestCase):

  def test_normal(self):
    self.assertEqual(util.bbox2roi([1, 2, 3, 4]), [2, 1, 6, 4])
    self.assertEqual(util.bbox2roi((1, 2, 3, 4)), [2, 1, 6, 4])

  def test_zeroDims(self):
    self.assertEqual(util.bbox2roi([1, 2, 0, 0]), [2, 1, 2, 1])

  def test_notSequence(self):
    with self.assertRaises(TypeError):
      util.bbox2roi(42)

  def test_LessThanFourNumbers(self):
    with self.assertRaises(ValueError):
      util.bbox2roi([42])

  def test_MoreThanFourNumbers(self):
    with self.assertRaises(ValueError):
      util.bbox2roi([42, 42, 42, 42, 42])

  def test_NotNumbers(self):
    with self.assertRaises(TypeError):
      util.bbox2roi(['a', 'b', 'c', 'd'])

  def test_negativeDims(self):
    with self.assertRaises(ValueError):
      util.bbox2roi([1, 2, 3, -1])


class TestRoi2Bbox (unittest.TestCase):

  def test_normal(self):
    self.assertEqual(util.roi2bbox([2, 1, 6, 4]), [1, 2, 3, 4])
    self.assertEqual(util.roi2bbox((2, 1, 6, 4)), [1, 2, 3, 4])

  def test_zeroDims(self):
    self.assertEqual(util.roi2bbox([2, 1, 2, 1]), [1, 2, 0, 0])

  def test_notSequence(self):
    with self.assertRaises(TypeError):
      util.roi2bbox(42)

  def test_LessThanFourNumbers(self):
    with self.assertRaises(ValueError):
      util.roi2bbox([42])

  def test_MoreThanFourNumbers(self):
    with self.assertRaises(ValueError):
      util.roi2bbox([42,42,42,42,42])

  def test_NotNumbers(self):
    with self.assertRaises(TypeError):
      util.roi2bbox(['a', 'b', 'c', 'd'])

  def test_negativeDims(self):
    with self.assertRaises(ValueError):
      util.roi2bbox([2, 1, 1, 2])


if __name__ == '__main__':
  logging.basicConfig (level=logging.ERROR)
  unittest.main()
