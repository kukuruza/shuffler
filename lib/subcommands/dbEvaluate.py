import os, os.path as op
import logging
import numpy as np
import cv2
import progressbar
import ast
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pprint
import PIL

from lib.backend import backendDb
from lib.backend import backendMedia
from lib.utils import util


def add_parsers(subparsers):
    evaluateDetectionParser(subparsers)
    evaluateSegmentationIoUParser(subparsers)
    evaluateBinarySegmentationParser(subparsers)


def _evaluateDetectionForClassPascal(c, c_gt, name, args):
    def _voc_ap(rec, prec):
        """ Compute VOC AP given precision and recall. """

        # First append sentinel values at the end.
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # Compute the precision envelope.
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # To calculate area under PR curve, look for points
        # where X axis (recall) changes value.
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Sum (\Delta recall) * prec.
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    c.execute('SELECT * FROM objects WHERE name=? ORDER BY score DESC',
              (name, ))
    entries_det = c.fetchall()
    logging.info('Total %d detected objects for class "%s"', len(entries_det),
                 name)

    # Go down dets and mark TPs and FPs.
    tp = np.zeros(len(entries_det), dtype=float)
    fp = np.zeros(len(entries_det), dtype=float)
    # Detected of no interest.
    ignored = np.zeros(len(entries_det), dtype=bool)

    # 'already_detected' used to penalize multiple detections of same GT box.
    already_detected = set()

    # Go through each detection.
    for idet, entry_det in enumerate(entries_det):

        bbox_det = np.array(backendDb.objectField(entry_det, 'bbox'),
                            dtype=float)
        imagefile = backendDb.objectField(entry_det, 'imagefile')
        name = backendDb.objectField(entry_det, 'name')

        # Get all GT boxes from the same imagefile [of the same class].
        c_gt.execute('SELECT * FROM objects WHERE imagefile=? AND name=?',
                     (imagefile, name))
        entries_gt = c_gt.fetchall()
        objectids_gt = [
            backendDb.objectField(entry, 'objectid') for entry in entries_gt
        ]
        bboxes_gt = np.array(
            [backendDb.objectField(entry, 'bbox') for entry in entries_gt],
            dtype=float)

        # Separately manage no GT boxes.
        if bboxes_gt.size == 0:
            fp[idet] = 1.
            continue

        # Intersection between bbox_det and all bboxes_gt.
        ixmin = np.maximum(bboxes_gt[:, 0], bbox_det[0])
        iymin = np.maximum(bboxes_gt[:, 1], bbox_det[1])
        ixmax = np.minimum(bboxes_gt[:, 0] + bboxes_gt[:, 2],
                           bbox_det[0] + bbox_det[2])
        iymax = np.minimum(bboxes_gt[:, 1] + bboxes_gt[:, 3],
                           bbox_det[1] + bbox_det[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        intersection = iw * ih

        # Union between bbox_det and all bboxes_gt.
        union = (bbox_det[2] * bbox_det[3] +
                 bboxes_gt[:, 2] * bboxes_gt[:, 3] - intersection)

        # IoU and get the best IoU.
        IoUs = intersection / union
        max_IoU = np.max(IoUs)
        objectid_gt = objectids_gt[np.argmax(IoUs)]
        logging.debug('max_IoU=%.3f for idet %d with objectid_gt %d.', max_IoU,
                      idet, objectid_gt)

        # Find which objects count towards TP and FN (should be detected).
        c_gt.execute(
            'SELECT * FROM objects WHERE imagefile=? AND name=? AND %s' %
            args.where_object_gt, (imagefile, name))
        entries_gt = c_gt.fetchall()
        objectids_gt_of_interest = [
            backendDb.objectField(entry, 'objectid') for entry in entries_gt
        ]

        # If 1) large enough IoU and
        #    2) this GT box was not detected before.
        if max_IoU > args.IoU_thresh and not objectid_gt in already_detected:
            if objectid_gt in objectids_gt_of_interest:
                tp[idet] = 1.
            else:
                ignored[idet] = True
            already_detected.add(objectid_gt)
        else:
            fp[idet] = 1.

    # Find the number of GT of interest.
    c_gt.execute(
        'SELECT COUNT(1) FROM objects WHERE %s AND name=?' %
        args.where_object_gt, (name, ))
    n_gt = c_gt.fetchone()[0]
    logging.info('Total objects of interest: %d', n_gt)

    # Remove dets, neither TP or FP.
    tp = tp[np.bitwise_not(ignored)]
    fp = fp[np.bitwise_not(ignored)]

    logging.info('ignored: %d, tp: %d, fp: %d, gt: %d',
                 np.count_nonzero(ignored), np.count_nonzero(tp),
                 np.count_nonzero(fp), n_gt)
    assert np.count_nonzero(tp) + np.count_nonzero(fp) + np.count_nonzero(
        ignored) == len(entries_det)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(n_gt)
    # Avoid divide by zero in case the first detection matches a difficult
    # ground truth.
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    aps = _voc_ap(rec, prec)
    print('Average precision for class "%s": %.4f' % (name, aps))
    return aps


def _writeCurveValues(out_dir, X, Y, metrics_name, name, header):
    if name is not None:
        name = util.validateFileName(name)
        stem = '%s-%s' % (metrics_name, name)
    else:
        stem = metrics_name
    plt.savefig(op.join(out_dir, '%s.png' % stem))
    plt.savefig(op.join(out_dir, '%s.eps' % stem))
    with open(op.join(out_dir, '%s.txt' % stem), 'w') as f:
        f.write('%s\n' % header)
        for x, y in zip(X, Y):
            f.write('%f %f\n' % (x, y))


def _beautifyPlot(ax):
    ax.grid(which='major', linewidth='0.5')
    ax.grid(which='minor', linewidth='0.2')
    loc = ticker.MultipleLocator(0.2)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    loc = ticker.MultipleLocator(0.1)
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.set_aspect('equal', adjustable='box')


def _evaluateDetectionForClassSklearn(c, c_gt, class_name, args, sklearn):
    ''' Helper function for evaluateDetection. '''

    # Detected objects sorted by descending score (confidence).
    if class_name is None:
        c.execute('SELECT * FROM objects ORDER BY score DESC')
    else:
        c.execute('SELECT * FROM objects WHERE name=? ORDER BY score DESC',
                  (class_name, ))
    entries_det = c.fetchall()
    logging.info('Num of positive "%s": %d', class_name, len(entries_det))

    # Create arrays 'y_score' with predicted scores, binary 'y_true' for GT,
    # and a binary 'y_ignored' for detected objects that are neither TP nor FP.
    y_score = np.zeros(len(entries_det), dtype=float)
    y_true = np.zeros(len(entries_det), dtype=bool)
    y_ignored = np.zeros(len(entries_det), dtype=bool)

    # 'already_detected' used to penalize multiple detections of same GT box
    already_detected = set()

    # Go through each detection.
    for idet, entry_det in enumerate(entries_det):

        bbox_det = np.array(backendDb.objectField(entry_det, 'bbox'),
                            dtype=float)
        imagefile = backendDb.objectField(entry_det, 'imagefile')
        name = backendDb.objectField(entry_det, 'name')
        score = backendDb.objectField(entry_det, 'score')

        y_score[idet] = score

        # Get all GT boxes from the same imagefile and of the same class.
        c_gt.execute('SELECT * FROM objects WHERE imagefile=? AND name=?',
                     (imagefile, name))
        entries_gt = c_gt.fetchall()
        objectids_gt = [
            backendDb.objectField(entry, 'objectid') for entry in entries_gt
        ]
        bboxes_gt = np.array(
            [backendDb.objectField(entry, 'bbox') for entry in entries_gt],
            dtype=float)

        # Separately manage the case of no GT boxes in this image.
        if bboxes_gt.size == 0:
            y_score[idet] = False
            continue

        # Intersection between bbox_det and all bboxes_gt.
        ixmin = np.maximum(bboxes_gt[:, 0], bbox_det[0])
        iymin = np.maximum(bboxes_gt[:, 1], bbox_det[1])
        ixmax = np.minimum(bboxes_gt[:, 0] + bboxes_gt[:, 2],
                           bbox_det[0] + bbox_det[2])
        iymax = np.minimum(bboxes_gt[:, 1] + bboxes_gt[:, 3],
                           bbox_det[1] + bbox_det[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        intersection = iw * ih

        # Union between bbox_det and all bboxes_gt.
        union = (bbox_det[2] * bbox_det[3] +
                 bboxes_gt[:, 2] * bboxes_gt[:, 3] - intersection)

        # Compute the best IoU between the bbox_det and all bboxes_gt.
        IoUs = intersection / union
        max_IoU = np.max(IoUs)
        objectid_gt = objectids_gt[np.argmax(IoUs)]
        logging.debug('max_IoU=%.3f for idet %d with objectid_gt %d.', max_IoU,
                      idet, objectid_gt)

        # Get all GT objects that are of interest.
        c_gt.execute(
            'SELECT * FROM objects WHERE imagefile=? AND name=? AND %s' %
            args.where_object_gt, (imagefile, name))
        entries_gt = c_gt.fetchall()
        objectids_gt_of_interest = [
            backendDb.objectField(entry, 'objectid') for entry in entries_gt
        ]

        # Compute TP and FP. An object is a TP if:
        #   1) it has a large enough IoU with a GT object and
        #   2) this GT object was not detected before.
        if max_IoU > args.IoU_thresh and not objectid_gt in already_detected:
            if objectid_gt not in objectids_gt_of_interest:
                y_ignored[idet] = True
            already_detected.add(objectid_gt)
            y_true[idet] = True
        else:
            y_true[idet] = False

    # It doesn't matter if y_ignore'd GT fall into TP or FP. Kick them out.
    y_score = y_score[np.bitwise_not(y_ignored)]
    y_true = y_true[np.bitwise_not(y_ignored)]

    # Find the number of GT of interest.
    if class_name is None:
        c_gt.execute('SELECT COUNT(1) FROM objects WHERE %s' %
                     args.where_object_gt)
    else:
        c_gt.execute(
            'SELECT COUNT(1) FROM objects WHERE %s AND name=?' %
            args.where_object_gt, (class_name, ))
    num_gt = c_gt.fetchone()[0]
    logging.info('Number of ground truth "%s": %d', class_name, num_gt)

    # Add FN to y_score and y_true.
    num_fn = num_gt - np.count_nonzero(y_true)
    logging.info('Number of false negative "%s": %d', class_name, num_fn)
    y_score = np.pad(y_score, [0, num_fn], constant_values=0.)
    y_true = np.pad(y_true, [0, num_fn], constant_values=True)

    # We need the point for threshold=0 to have y=0. Not sure why it's not yet.
    # TODO: figure out how to do it properly.
    y_score = np.pad(y_score, [0, 1000000], constant_values=0.0001)
    y_true = np.pad(y_true, [0, 1000000], constant_values=False)

    if 'precision_recall_curve' in args.extra_metrics:
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            y_true=y_true, probas_pred=y_score)
        if args.out_dir:
            plt.clf()
            plt.plot(recall, precision)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            _beautifyPlot(plt.gca())
            _writeCurveValues(args.out_dir, recall, precision,
                              'precision-recall', class_name,
                              'recall precision')

    if 'roc_curve' in args.extra_metrics:
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=y_true,
                                                probas_pred=y_score)
        sklearn.metrics.auc(x=fpr, y=tpr)
        if args.out_dir:
            plt.clf()
            plt.plot(fpr, tpr)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            _beautifyPlot(plt.gca())
            _writeCurveValues(args.out_dir, fpr, tpr, 'roc', class_name,
                              'fpr tpr')

    # Compute all metrics for this class.
    aps = sklearn.metrics.average_precision_score(y_true=y_true,
                                                  y_score=y_score)
    if class_name is None:
        print('Average precision: %.4f' % aps)
    else:
        print('Average precision for class "%s": %.4f' % (class_name, aps))
    return aps


def evaluateDetectionParser(subparsers):
    parser = subparsers.add_parser(
        'evaluateDetection',
        description='Evaluate detections given a ground truth database.')
    parser.set_defaults(func=evaluateDetection)
    parser.add_argument('--gt_db_file', required=True)
    parser.add_argument('--IoU_thresh', type=float, default=0.5)
    parser.add_argument('--where_object_gt', default='TRUE')
    parser.add_argument(
        '--out_dir',
        help='If specified, plots and text files are written here.')
    parser.add_argument(
        '--extra_metrics',
        nargs='+',
        default=[],
        choices=[
            'precision_recall_curve',
            'roc_curve',
        ],
        help='Select metrics to be computed in addition to average precision. '
        'This is implemented only for evaluation_backend="sklearn". '
        'They are computed for every class. The names match those at '
        'https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics'
    )
    parser.add_argument(
        '--evaluation_backend',
        choices=['sklearn', 'pascal-voc', 'sklearn-all-classes'],
        default='sklearn',
        help='Detection evaluation is different across papers and methods. '
        'PASCAL VOC produces average-precision score a bit different '
        'than the sklearn package. A good overview on metrics: '
        'https://github.com/rafaelpadilla/Object-Detection-Metrics. '
        '"sklearn-all-classes" reports only one accuracy.')


def evaluateDetection(c, args):
    if 'sklearn' in args.evaluation_backend:
        import sklearn.metrics

    # Load the ground truth database.
    if not op.exists(args.gt_db_file):
        raise FileNotFoundError('File does not exist: %s' % args.gt_db_file)
    conn_gt = backendDb.connect(args.gt_db_file, 'load_to_memory')
    c_gt = conn_gt.cursor()

    # Some info for logging.
    c.execute('SELECT COUNT(1) FROM objects')
    logging.info('The evaluated database has %d objects.', c.fetchone()[0])
    c_gt.execute('SELECT COUNT(1) FROM objects WHERE %s' %
                 args.where_object_gt)
    logging.info('The ground truth database has %d objects of interest.',
                 c_gt.fetchone()[0])

    c_gt.execute('SELECT DISTINCT(name) FROM objects')
    names = c_gt.fetchall()
    if args.evaluation_backend == 'sklearn':
        for name, in names:
            _evaluateDetectionForClassSklearn(c, c_gt, name, args, sklearn)
    elif args.evaluation_backend == 'pascal-voc':
        for name, in names:
            if args.metrics is not None:
                logging.warning('extra_metrics not supported for pascal-voc.')
            _evaluateDetectionForClassPascal(c, c_gt, name, args)
    elif args.evaluation_backend == 'sklearn-all-classes':
        # This method does not separate results by classes.
        _evaluateDetectionForClassSklearn(c, c_gt, None, args, sklearn)
    else:
        assert False
    conn_gt.close()


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k],
                       minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def calc_fw_iu(hist):
    pred_per_class = hist.sum(0)
    gt_per_class = hist.sum(1)
    return np.nansum(
        (gt_per_class * np.diag(hist)) /
        (pred_per_class + gt_per_class - np.diag(hist))) / gt_per_class.sum()


def calc_pixel_accuracy(hist):
    gt_per_class = hist.sum(1)
    return np.diag(hist).sum() / gt_per_class.sum()


def calc_mean_accuracy(hist):
    gt_per_class = hist.sum(1)
    acc_per_class = np.diag(hist) / gt_per_class
    return np.nanmean(acc_per_class)


def save_colorful_images(prediction, filename, palette, postfix='_color.png'):
    im = PIL.Image.fromarray(palette[prediction.squeeze()])
    im.save(filename[:-4] + postfix)


def label_mapping(input_, mapping):
    output = np.copy(input_)
    for ind in range(len(mapping)):
        output[input_ == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def plot_confusion_matrix(cm, classes, normalize=False, cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if cmap is None:
        cmap = plt.cm.Blues

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logging.info("Normalized confusion matrix.")
    else:
        logging.info(
            'Confusion matrix will be computed without normalization.')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted label')


def _label2classMapping(gt_mapping_dict, pred_mapping_dict):
    ''' Parse user-defined label mapping dictionaries. '''

    # "gt_mapping_dict" maps mask pixel-values to classes.
    labelmap_gt = ast.literal_eval(gt_mapping_dict)
    labelmap_pr = ast.literal_eval(
        pred_mapping_dict) if pred_mapping_dict else labelmap_gt
    # Create a list of classes.
    class_names = list(labelmap_gt.values())
    labelmap_gt_new = {}
    # Here, we remap pixel-values to indices of class_names.
    for key in labelmap_gt:
        labelmap_gt_new[key] = class_names.index(labelmap_gt[key])
    labelmap_gt = labelmap_gt_new
    labelmap_pr_new = {}
    for key in labelmap_pr:
        if not labelmap_pr[key] in class_names:
            raise ValueError(
                'Class %s is in "pred_mapping_dict" but not in "gt_mapping_dict"'
            )
        labelmap_pr_new[key] = class_names.index(labelmap_pr[key])
    labelmap_pr = labelmap_pr_new
    return labelmap_gt, labelmap_pr, class_names


def evaluateSegmentationIoUParser(subparsers):
    parser = subparsers.add_parser(
        'evaluateSegmentationIoU',
        description='Evaluate mask segmentation w.r.t. a ground truth db.')
    parser.set_defaults(func=evaluateSegmentationIoU)
    parser.add_argument('--gt_db_file', required=True)
    parser.add_argument('--where_image', default='TRUE')
    parser.add_argument(
        '--out_dir',
        help='If specified, output files with be written to "out_dir".')
    parser.add_argument(
        '--out_prefix',
        default='',
        help='A prefix to add to output filenames, '
        'Use it to keep predictions from different epochs in one dir.')
    parser.add_argument(
        '--gt_mapping_dict',
        required=True,
        help=
        'A map from ground truth maskfile to classes written as a json string. '
        'E.g. "{0: \'background\', 255: \'car\'}"')
    parser.add_argument(
        '--pred_mapping_dict',
        help='A map from predicted masks to classes written as a json string, '
        'if different from "gt_mapping_dict"')
    parser.add_argument(
        '--class_to_record_iou',
        help='If specified, IoU for a class is recorded into the "score" '
        'field of the "images" table. '
        'If not specified, mean IoU is recorded. '
        'Should correspond to values of "gt_mapping_dict". E.g. "background".')
    parser.add_argument(
        '--out_summary_file',
        help='Text file, where the summary is going to be appended as just one '
        'line of format: out_prefix \\t IoU_class1 \\t IoU_class2 \\t etc.')


def evaluateSegmentationIoU(c, args):
    import pandas as pd
    import matplotlib.pyplot as plt

    # Get corresponding maskfiles from predictions and ground truth.
    logging.info('Opening ground truth dataset: %s', args.gt_db_file)
    c.execute('ATTACH ? AS "attached"', (args.gt_db_file, ))
    c.execute('SELECT pr.imagefile,pr.maskfile,gt.maskfile '
              'FROM images pr INNER JOIN attached.images gt '
              'WHERE pr.imagefile=gt.imagefile AND pr.maskfile IS NOT NULL '
              'AND gt.maskfile IS NOT NULL '
              'AND %s '
              'ORDER BY pr.imagefile ASC' % args.where_image)
    entries = c.fetchall()
    logging.info(
        'Total %d images in both the open and the ground truth databases.',
        len(entries))
    logging.debug(pprint.pformat(entries))

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    labelmap_gt, labelmap_pr, class_names = _label2classMapping(
        args.gt_mapping_dict, args.pred_mapping_dict)

    if args.class_to_record_iou is not None and not args.class_to_record_iou in class_names:
        raise ValueError(
            'class_to_record_iou=%s is not among values of gt_mapping_dict=%s'
            % (args.class_to_record_iou, args.gt_mapping_dict))

    hist_all = np.zeros((len(class_names), len(class_names)))

    for imagefile, maskfile_pr, maskfile_gt in progressbar.progressbar(
            entries):

        # Load masks and bring them to comparable form.
        mask_gt = util.applyLabelMappingToMask(imreader.maskread(maskfile_gt),
                                               labelmap_gt)
        mask_pr = util.applyLabelMappingToMask(imreader.maskread(maskfile_pr),
                                               labelmap_pr)
        mask_pr = cv2.resize(mask_pr, (mask_gt.shape[1], mask_gt.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        # Evaluate one image pair.
        careabout = ~np.isnan(mask_gt)
        mask_gt = mask_gt[careabout][:].astype(int)
        mask_pr = mask_pr[careabout][:].astype(int)
        hist = fast_hist(mask_gt, mask_pr, len(class_names))
        hist_all += hist

        # Compute and record results by image.
        iou_list = per_class_iu(hist)
        if args.class_to_record_iou is None:
            iou = iou_list.mean()
        else:
            iou = iou_list[class_names.index(args.class_to_record_iou)]
        c.execute('UPDATE images SET score=? WHERE imagefile=?',
                  (iou, imagefile))

    # Get label distribution.
    pr_per_class = hist_all.sum(0)
    gt_per_class = hist_all.sum(1)

    iou_list = per_class_iu(hist_all)
    fwIoU = calc_fw_iu(hist_all)
    pixAcc = calc_pixel_accuracy(hist_all)
    mAcc = calc_mean_accuracy(hist_all)

    result_df = pd.DataFrame({
        'class': class_names,
        'IoU': iou_list,
        "pr_distribution": pr_per_class,
        "gt_distribution": gt_per_class,
    })
    result_df["IoU"] *= 100  # Changing to percent ratio.

    result_df.set_index("class", inplace=True)
    print("---- info per class -----")
    print(result_df)

    result_ser = pd.Series({
        "pixAcc": pixAcc,
        "mAcc": mAcc,
        "fwIoU": fwIoU,
        "mIoU": iou_list.mean()
    })
    result_ser = result_ser[["pixAcc", "mAcc", "fwIoU", "mIoU"]]
    result_ser *= 100  # change to percent ratio

    if args.out_dir is not None:
        if not op.exists(args.out_dir):
            os.makedirs(args.out_dir)

        out_summary_path = op.join(args.out_dir, args.out_summary_file)
        logging.info('Will add summary to: %s', out_summary_path)
        with open(out_summary_path, 'a') as f:
            f.write(args.out_prefix + '\t' +
                    '\t'.join(['%.2f' % x for x in result_df['IoU']]) + '\n')

        # Save confusion matrix
        fig = plt.figure()
        normalized_hist = (hist.astype("float") /
                           hist.sum(axis=1)[:, np.newaxis])

        plot_confusion_matrix(normalized_hist, classes=class_names)
        outfigfn = op.join(args.out_dir, "%sconf_mat.pdf" % args.out_prefix)
        fig.savefig(outfigfn,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=300)
        print("Confusion matrix was saved to %s" % outfigfn)

        outdffn = op.join(args.out_dir,
                          "%seval_result_df.csv" % args.out_prefix)
        result_df.to_csv(outdffn)
        print('Info per class was saved at %s !' % outdffn)
        outserfn = op.join(args.out_dir,
                           "%seval_result_ser.csv" % args.out_prefix)
        result_ser.to_csv(outserfn)
        print('Total result is saved at %s !' % outserfn)


def getPrecRecall(tp, fp, fn):
    ''' Accumulate into Precision-Recall curve. '''
    ROC = np.zeros((256, 2), dtype=float)
    for val in range(256):
        if tp[val] == 0 and fp[val] == 0:
            precision = -1.
        else:
            precision = tp[val] / float(tp[val] + fp[val])
        if tp[val] == 0 and fn[val] == 0:
            recall = -1.
        else:
            recall = tp[val] / float(tp[val] + fn[val])
        ROC[val, 0] = recall
        ROC[val, 1] = precision
    ROC = ROC[np.bitwise_and(ROC[:, 0] != -1, ROC[:, 1] != -1), :]
    ROC = np.vstack((ROC, np.array([0, ROC[-1, 1]])))
    area = -np.trapz(x=ROC[:, 0], y=ROC[:, 1])
    return ROC, area


def evaluateBinarySegmentationParser(subparsers):
    parser = subparsers.add_parser(
        'evaluateBinarySegmentation',
        description=
        'Evaluate mask segmentation ROC curve w.r.t. a ground truth db. '
        'Ground truth values must be 0 for background, 255 for foreground, '
        'and the rest for "dontcare".'
        'Predicted mask must be grayscale in [0,255], '
        'with brightness meaning probability of foreground.')
    parser.set_defaults(func=evaluateBinarySegmentation)
    parser.add_argument('--gt_db_file', required=True)
    parser.add_argument('--where_image', default='TRUE')
    parser.add_argument(
        '--out_dir',
        help='If specified, result files with be written to "out_dir".')
    parser.add_argument(
        '--out_prefix',
        default='',
        help='A prefix to add to output filenames, '
        'Use it to keep predictions from different epochs in one dir.')
    parser.add_argument('--display_images_roc',
                        action='store_true',
                        help='Specify to display on screen')


def evaluateBinarySegmentation(c, args):
    import pandas as pd

    # Get corresponding maskfiles from predictions and ground truth.
    c.execute('ATTACH ? AS "attached"', (args.gt_db_file, ))
    c.execute('SELECT pr.imagefile,pr.maskfile,gt.maskfile '
              'FROM images pr INNER JOIN attached.images gt '
              'WHERE pr.imagefile=gt.imagefile '
              'AND pr.maskfile IS NOT NULL '
              'AND gt.maskfile IS NOT NULL '
              'AND %s '
              'ORDER BY pr.imagefile ASC' % args.where_image)
    entries = c.fetchall()
    logging.info(
        'Total %d images in both the open and the ground truth databases.' %
        len(entries))
    logging.debug(pprint.pformat(entries))

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    TPs = np.zeros((256, ), dtype=int)
    FPs = np.zeros((256, ), dtype=int)
    FNs = np.zeros((256, ), dtype=int)

    if args.display_images_roc:
        fig = plt.figure()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    for imagefile, maskfile_pr, maskfile_gt in progressbar.progressbar(
            entries):

        # Load masks and bring them to comparable form.
        mask_gt = imreader.maskread(maskfile_gt)
        mask_pr = imreader.maskread(maskfile_pr)
        mask_pr = cv2.resize(mask_pr, (mask_gt.shape[1], mask_gt.shape[0]),
                             cv2.INTER_NEAREST)

        # Some printputs.
        gt_pos = np.count_nonzero(mask_gt == 255)
        gt_neg = np.count_nonzero(mask_gt == 0)
        gt_other = mask_gt.size - gt_pos - gt_neg
        logging.debug('GT: positive: %d, negative: %d, others: %d.', gt_pos,
                      gt_neg, gt_other)

        # If there is torch.
        try:
            import torch
            # Use only relevant pixels (not the 'dontcare' class.)
            relevant = np.bitwise_or(mask_gt == 0, mask_gt == 255)
            mask_gt = mask_gt[relevant].flatten()
            mask_pr = mask_pr[relevant].flatten()
            mask_gt = torch.Tensor(mask_gt)
            mask_pr = torch.Tensor(mask_pr)
            try:
                mask_gt = mask_gt.cuda()
                mask_pr = mask_pr.cuda()
            except RuntimeError:
                pass

            TP = np.zeros((256, ), dtype=int)
            FP = np.zeros((256, ), dtype=int)
            FN = np.zeros((256, ), dtype=int)
            for val in range(256):
                tp = torch.nonzero(torch.mul(mask_pr > val,
                                             mask_gt == 255)).size()[0]
                fp = torch.nonzero(torch.mul(mask_pr > val,
                                             mask_gt != 255)).size()[0]
                fn = torch.nonzero(torch.mul(mask_pr <= val,
                                             mask_gt == 255)).size()[0]
                tn = torch.nonzero(torch.mul(mask_pr <= val,
                                             mask_gt != 255)).size()[0]
                TP[val] = tp
                FP[val] = fp
                FN[val] = fn
                TPs[val] += tp
                FPs[val] += fp
                FNs[val] += fn
            ROC, area = getPrecRecall(TP, FP, FN)
            logging.info('%s\t%.2f' % (op.basename(imagefile), area * 100.))

        except ImportError:
            # TODO: write the same without torch, on CPU
            raise NotImplementedError(
                'Non-torch implementation is still to be implemented.')

        if args.display_images_roc:
            plt.plot(ROC[:, 0], ROC[:, 1], 'go-', linewidth=2, markersize=4)
            plt.pause(0.05)
            fig.show()

    # Accumulate into Precision-Recall curve.
    ROC, area = getPrecRecall(TPs, FPs, FNs)
    print(
        "Average across image area under the Precision-Recall curve, perc: %.2f"
        % (area * 100.))

    if args.out_dir is not None:
        if not op.exists(args.out_dir):
            os.makedirs(args.out_dir)
        fig = plt.figure()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot(ROC[:, 0], ROC[:, 1], 'bo-', linewidth=2, markersize=6)
        out_plot_path = op.join(args.out_dir,
                                '%srecall-prec.png' % args.out_prefix)
        fig.savefig(out_plot_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=300)
