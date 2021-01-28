import sqlite3
import argparse
import numpy as np
import sklearn.metrics
import progressbar
import nose

from lib.backend import backendDb
from lib.subcommands import dbEvaluate
from lib.utils import testUtils


class Test_evaluateDetectionForClass_Base(testUtils.Test_emptyDb):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)
        self.conn_gt = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn_gt)
        self.args = argparse.Namespace(IoU_thresh=0.5,
                                       where_object_gt='TRUE',
                                       extra_metrics=[])

    def tearDown(self):
        self.conn.close()
        self.conn_gt.close()


class Test_evaluateDetectionForClass_0gt(Test_evaluateDetectionForClass_Base):
    def test_0gt_0det(self):
        ''' 0 gt, 0 det. Pascal AP should be 0. Sklearn AP should be NaN. '''
        c = self.conn.cursor()
        c_gt = self.conn_gt.cursor()
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps, 0.)
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertTrue(np.isnan(aps))

    def test_0gt_1fp(self):
        ''' 0 gt, 1 fp. AP should be NaN. '''
        c = self.conn.cursor()
        # Add 1 FP object.
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,40,30,"name",1.0)')
        c_gt = self.conn_gt.cursor()
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertTrue(np.isnan(aps))
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertTrue(np.isnan(aps))


class Test_evaluateDetectionForClass_1gt(Test_evaluateDetectionForClass_Base):
    def setUp(self):
        super(Test_evaluateDetectionForClass_1gt, self).setUp()
        c_gt = self.conn_gt.cursor()
        c_gt.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c_gt.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,30,10,"name",1.0)')
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')

    def test_1tp(self):
        ''' 1 tp. AP should be 1. '''
        c = self.conn.cursor()
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,30,10,"name",1.0)')
        c_gt = self.conn_gt.cursor()
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps, 1.)
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertEqual(aps, 1.)

    def test_1tp_IoU(self):
        '''
        Check different IoU_thresholds.
        1 tp. AP should be 1. IoU with GT = 0.5.
        '''
        c = self.conn.cursor()
        # Shifted by dx=10 (given width=30)
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40+10,20,30,10,"name",1.0)')
        c_gt = self.conn_gt.cursor()
        # IoU_thresh=0.49.
        self.args.IoU_thresh = 0.49
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps, 1., msg='pascal, IoU_thresh=0.49.')
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertAlmostEqual(aps, 1., places=4, msg='sklearn, IoU_thr=0.49.')
        # IoU_thresh=0.51.
        self.args.IoU_thresh = 0.51
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps, 0., msg='pascal, IoU_thresh=0.51.')
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertAlmostEqual(aps, 0., places=4, msg='sklearn, IoU_thr=0.51.')

    def test_1fn(self):
        ''' pascal, 1 fn. AP should be 0. '''
        c = self.conn.cursor()
        c_gt = self.conn_gt.cursor()
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps, 0.)
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertAlmostEqual(aps, 0., places=4)

    def test_1fp_1fn(self):
        ''' 1 fp, 1 fn. AP should be 0. '''
        c = self.conn.cursor()
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,-40,20,30,10,"name",1.0)')
        c_gt = self.conn_gt.cursor()
        aps_pascal = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps_pascal, 0.)
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertAlmostEqual(aps, 0., places=4)


class Test_evaluateDetectionForClass_2gt(Test_evaluateDetectionForClass_Base):
    def setUp(self):
        super(Test_evaluateDetectionForClass_2gt, self).setUp()
        c_gt = self.conn_gt.cursor()
        c_gt.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c_gt.execute('INSERT INTO images(imagefile) VALUES ("image1")')
        c_gt.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image0",0,40,20,40,20,"name")')
        c_gt.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image1",1,40,20,40,20,"name")')
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute('INSERT INTO images(imagefile) VALUES ("image1")')

    def test_1tp_whereObjectsGt(self):
        ''' 1 tp, using where_gt_objects to keep only 1 GT. AP should be 1. '''
        c = self.conn.cursor()
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,40,20,"name",0.5)')
        c_gt = self.conn_gt.cursor()
        self.args.where_object_gt = 'imagefile="image0"'
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps, 1)
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertAlmostEqual(aps, 1, places=4)

    def test_1tp_1fn(self):
        ''' 1 tp, 1 fn. prec(0.5)=1, prec(1)=0. AP should be 0.5. '''
        c = self.conn.cursor()
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,40,20,"name",0.5)')
        c_gt = self.conn_gt.cursor()
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps, 0.5)
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertAlmostEqual(aps, 0.5, places=4)

    def test_1tp_1fn_1fp(self):
        '''
        1 tp, 1 fn, 1 fp. prec(0.5)=0.5, prec(1)=0.
        Sklearn: AP should be 0.25.
        Pascal:  AP should be 0.5 (not sure why.)
        '''
        c = self.conn.cursor()
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,40,20,"name",0.5)')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",1,140,20,40,20,"name",0.5)')
        c_gt = self.conn_gt.cursor()
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps, 0.5)
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertAlmostEqual(aps, 0.25, places=4)  # Not sure why 0.25.

    def test_2tp_2fp_atSeveralThresh(self):
        '''
        1 tp, @thres=0.2.
        1 tp, 2 fp @thres=0.1;
        prec(0.5)=1, prec(1)=0.5.
        Sklearn: AP should be 1 (don't know why)
        Pascal:  AP should be 0.75.
        '''
        c = self.conn.cursor()
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,40,20,"name",0.2)')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image1",1,40,20,40,20,"name",0.1)')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",2,140,20,40,20,"name",0.1)')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",3,140,20,40,20,"name",0.1)')
        c_gt = self.conn_gt.cursor()
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertEqual(aps, 1.0, msg='pascal')  # Not sure why 1.0.
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertAlmostEqual(aps, 0.75, places=4, msg='sklearn')


class Test_evaluateDetectionForClass_Ngt(Test_evaluateDetectionForClass_Base):
    def setUp(self):
        super(Test_evaluateDetectionForClass_Ngt, self).setUp()
        self.conn_gt = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn_gt)
        c_gt = self.conn_gt.cursor()
        self.N = 1000
        for i in range(self.N):
            imagefile = 'image%d' % i
            c_gt.execute('INSERT INTO images(imagefile) VALUES (?)',
                         (imagefile, ))
            c_gt.execute(
                'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
                'VALUES (?,?,40,20,40,20,"name")', (imagefile, i))

    def test_precRecallIsTriangle(self):
        '''
        Precision-recall plot is a triangle made with self.N different thresholds.
        At each threshold one fn becomes a tp, and one fp is added.
        '''
        c_gt = self.conn_gt.cursor()
        c = self.conn.cursor()
        for i in range(self.N):
            imagefile = 'image%d' % i
            c.execute('INSERT INTO images(imagefile) VALUES (?)',
                      (imagefile, ))
            c.execute(
                'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
                'VALUES (?,?,40,20,40,20,"name",?)',
                (imagefile, 2 * i, 1. - i / self.N))
            c.execute(
                'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
                'VALUES (?,?,140,20,40,20,"name",?)',
                (imagefile, 2 * i + 1, 1. - i / self.N))
        aps = dbEvaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', self.args)
        self.assertAlmostEqual(aps, 0.5, places=1, msg='pascal')
        aps = dbEvaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', self.args, sklearn)
        self.assertAlmostEqual(aps, 0.5, places=1, msg='sklearn')


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
