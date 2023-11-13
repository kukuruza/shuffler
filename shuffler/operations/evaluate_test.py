import pytest
import sqlite3
import argparse
import numpy as np
import sklearn.metrics

from shuffler.backend import backend_db
from shuffler.operations import evaluate
from shuffler.utils import testing as testing_utils


class Test_EvaluateDetectionForClass_Base(testing_utils.EmptyDb):
    @pytest.fixture()
    def empty_c_gt(self):
        conn_gt = sqlite3.connect(':memory:')
        backend_db.createDb(conn_gt)
        yield conn_gt.cursor()
        conn_gt.close()

    @pytest.fixture()
    def args(self):
        return argparse.Namespace(IoU_thresh=0.5,
                                  where_object_gt='TRUE',
                                  extra_metrics=[],
                                  evaluation_backend='by-class')


class Test_EvaluateDetectionForClass_0gt(Test_EvaluateDetectionForClass_Base):
    @pytest.mark.filterwarnings('ignore::UserWarning:sklearn')
    def test_0gt_0det(self, c, empty_c_gt, args):
        ''' 0 gt, 0 det. Pascal and Sklearn AP should be 0. '''
        c_gt = empty_c_gt
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 0., 'Pascal: %s' % aps
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        assert aps == 0., 'Sklearn: %s' % aps

    @pytest.mark.filterwarnings('ignore::UserWarning:sklearn')
    def test_0gt_1fp(self, c, empty_c_gt, args):
        ''' 0 gt, 1 fp. AP should be NaN. '''
        c_gt = empty_c_gt
        # Add 1 FP object.
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,40,30,"name",1.0)')
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 0, 'Pascal: %s' % aps
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        assert aps == 0., 'Sklearn: %s' % aps


class Test_EvaluateDetectionForClass_1gt(Test_EvaluateDetectionForClass_Base):
    @pytest.fixture()
    def c_and_c_gt(self, conn, empty_c_gt):
        c = conn.cursor()
        c_gt = empty_c_gt
        c_gt.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c_gt.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,30,10,"name",1.0)')
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        yield c, c_gt

    def test_1tp(self, c_and_c_gt, args):
        ''' 1 tp. AP should be 1. '''
        c, c_gt = c_and_c_gt
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,30,10,"name",1.0)')
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 1.
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        assert aps == 1.

    def test_1tp_IoU(self, c_and_c_gt, args):
        '''
        Check different IoU_thresholds.
        1 tp. AP should be 1. IoU with GT = 0.5.
        '''
        c, c_gt = c_and_c_gt
        # Shifted by dx=10 (given width=30)
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40+10,20,30,10,"name",1.0)')
        # IoU_thresh=0.49.
        args.IoU_thresh = 0.49
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 1., 'pascal, IoU_thresh=0.49.'
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        np.testing.assert_almost_equal(aps,
                                       1.,
                                       decimal=4,
                                       err_msg='sklearn, IoU_thr=0.49.')
        # IoU_thresh=0.51.
        args.IoU_thresh = 0.51
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 0., 'pascal, IoU_thresh=0.51.'
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        np.testing.assert_almost_equal(aps,
                                       0.,
                                       decimal=4,
                                       err_msg='sklearn, IoU_thr=0.51.')

    def test_1fn(self, c_and_c_gt, args):
        ''' pascal, 1 fn. AP should be 0. '''
        c, c_gt = c_and_c_gt
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 0.
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        np.testing.assert_almost_equal(aps, 0., decimal=4)

    def test_1fp_1fn(self, c_and_c_gt, args):
        ''' 1 fp, 1 fn. AP should be 0. '''
        c, c_gt = c_and_c_gt
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,-40,20,30,10,"name",1.0)')
        aps_pascal = evaluate._evaluateDetectionForClassPascal(
            c, c_gt, 'name', args)
        assert aps_pascal == 0.
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        np.testing.assert_almost_equal(aps, 0., decimal=4)


class Test_EvaluateDetectionForClass_2gt(Test_EvaluateDetectionForClass_Base):
    @pytest.fixture()
    def c_and_c_gt(self, conn, empty_c_gt):
        c_gt = empty_c_gt
        c_gt.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c_gt.execute('INSERT INTO images(imagefile) VALUES ("image1")')
        c_gt.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image0",0,40,20,40,20,"name")')
        c_gt.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
            'VALUES ("image1",1,40,20,40,20,"name")')

        c = conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute('INSERT INTO images(imagefile) VALUES ("image1")')
        yield c, c_gt

    def test_1tp_whereObjectsGt(self, c_and_c_gt, args):
        ''' 1 tp, using where_gt_objects to keep only 1 GT. AP should be 1. '''
        c, c_gt = c_and_c_gt
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,40,20,"name",0.5)')
        args.where_object_gt = 'imagefile="image0"'
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 1
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        np.testing.assert_almost_equal(aps, 1, decimal=4)

    def test_1tp_1fn(self, c_and_c_gt, args):
        ''' 1 tp, 1 fn. prec(0.5)=1, prec(1)=0. AP should be 0.5. '''
        c, c_gt = c_and_c_gt
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,40,20,"name",0.5)')
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 0.5
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        np.testing.assert_almost_equal(aps, 0.5, decimal=4)

    def test_1tp_1fn_1fp(self, c_and_c_gt, args):
        '''
        1 tp, 1 fn, 1 fp. prec(0.5)=0.5, prec(1)=0.
        Sklearn: AP should be 0.25.
        Pascal:  AP should be 0.5 (not sure why.)
        '''
        c, c_gt = c_and_c_gt
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",0,40,20,40,20,"name",0.5)')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
            'VALUES ("image0",1,140,20,40,20,"name",0.5)')
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 0.5
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        np.testing.assert_almost_equal(aps, 0.25,
                                       decimal=4)  # Not sure why 0.25.

    def test_2tp_2fp_atSeveralThresh(self, c_and_c_gt, args):
        '''
        1 tp, @thres=0.2.
        1 tp, 2 fp @thres=0.1;
        prec(0.5)=1, prec(1)=0.5.
        Sklearn: AP should be 1 (don't know why)
        Pascal:  AP should be 0.75.
        '''
        c, c_gt = c_and_c_gt
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
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        assert aps == 1.0, 'pascal'  # Not sure why 1.0.
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        np.testing.assert_almost_equal(aps, 0.75, decimal=4, err_msg='sklearn')


class Test_EvaluateDetectionForClass_Ngt(Test_EvaluateDetectionForClass_Base):
    @pytest.fixture()
    def c_gt_and_N(self, empty_c_gt):
        c_gt = empty_c_gt
        N = 1000
        for i in range(N):
            imagefile = 'image%d' % i
            c_gt.execute('INSERT INTO images(imagefile) VALUES (?)',
                         (imagefile, ))
            c_gt.execute(
                'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name) '
                'VALUES (?,?,40,20,40,20,"name")', (imagefile, i))
        yield c_gt, N

    def test_prec_recall_is_triangle(self, c, c_gt_and_N, args):
        '''
        Precision-recall plot is a triangle made with N different thresholds.
        At each threshold one fn becomes a tp, and one fp is added.
        '''
        c_gt, N = c_gt_and_N
        for i in range(N):
            imagefile = 'image%d' % i
            c.execute('INSERT INTO images(imagefile) VALUES (?)',
                      (imagefile, ))
            c.execute(
                'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
                'VALUES (?,?,40,20,40,20,"name",?)',
                (imagefile, 2 * i, 1. - i / N))
            c.execute(
                'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) '
                'VALUES (?,?,140,20,40,20,"name",?)',
                (imagefile, 2 * i + 1, 1. - i / N))
        aps = evaluate._evaluateDetectionForClassPascal(c, c_gt, 'name', args)
        np.testing.assert_almost_equal(aps, 0.5, decimal=1, err_msg='pascal')
        aps = evaluate._evaluateDetectionForClassSklearn(
            c, c_gt, 'name', args, sklearn)
        np.testing.assert_almost_equal(aps, 0.5, decimal=1, err_msg='sklearn')
