import os.path as op
import shutil
import unittest
import tempfile
import progressbar
import nose

from lib.utils import util


class Test_CopyWithBackup(unittest.TestCase):
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


class Test_getIntersectingObjects(unittest.TestCase):
    def test_empty(self):
        pairs_to_merge = util.getIntersectingObjects([], [], 0.5)
        self.assertEqual(pairs_to_merge, [])

    def test_firstEmpty(self):
        objects2 = [(1, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = util.getIntersectingObjects([], objects2, 0.5)
        self.assertEqual(pairs_to_merge, [])

    def test_identical(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = util.getIntersectingObjects(objects1, objects2, 0.5)
        self.assertEqual(pairs_to_merge, [(1, 2)])

    def test_identical_sameId(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(1, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = util.getIntersectingObjects(objects1,
                                                     objects2,
                                                     0.5,
                                                     same_id_ok=False)
        self.assertEqual(pairs_to_merge, [])

    def test_nonIntersecting(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 20, 20, 40, 40, 'name2', 1.)]
        pairs_to_merge = util.getIntersectingObjects(objects1, objects2, 0.5)
        self.assertEqual(pairs_to_merge, [])

    def test_twoIntersecting(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 20, 20, 30, 30, 'name2', 1.),
                    (3, 'image', 10, 10, 30, 30, 'name3', 1.)]
        pairs_to_merge = util.getIntersectingObjects(objects1, objects2, 0.1)
        self.assertEqual(pairs_to_merge, [(1, 3)])

    def test_twoAndTwoIntersecting(self):
        # #1.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 20, 20, 30, 30, 'name3', 1.),
                    (4, 'image', 10, 10, 30, 30, 'name4', 1.)]
        pairs_to_merge = util.getIntersectingObjects(objects1, objects2, 0.1)
        self.assertEqual(set(pairs_to_merge), set([(1, 4), (2, 3)]))
        # #2.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 20, 20, 30, 30, 'name4', 1.)]
        pairs_to_merge = util.getIntersectingObjects(objects1, objects2, 0.1)
        self.assertEqual(set(pairs_to_merge), set([(1, 3), (2, 4)]))
        # #3.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 0, 0, 30, 30, 'name4', 1.),
                    (5, 'image', 20, 20, 30, 30, 'name5', 1.)]
        pairs_to_merge = util.getIntersectingObjects(objects1, objects2, 0.1)
        self.assertEqual(set(pairs_to_merge), set([(1, 3), (2, 5)]))


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
