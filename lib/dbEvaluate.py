import os, os.path as op
import sys
import logging
import sqlite3
import numpy as np
import cv2
from progressbar import progressbar
import simplejson as json
from ast import literal_eval
from matplotlib import pyplot as plt
from pprint import pformat

from .backendDb import objectField
from .backendMedia import MediaReader
from .util import applyLabelMappingToMask



def add_parsers(subparsers):
  evaluateDetectionParser(subparsers)
  evaluateSegmentationIoUParser(subparsers)
  evaluateBinarySegmentationParser(subparsers)



def _voc_ap(rec, prec):
  """ Compute VOC AP given precision and recall. """

  # first append sentinel values at the end
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))

  # compute the precision envelope
  for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

  # to calculate area under PR curve, look for points
  # where X axis (recall) changes value
  i = np.where(mrec[1:] != mrec[:-1])[0]

  # and sum (\Delta recall) * prec
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

def evaluateDetectionParser(subparsers):
  parser = subparsers.add_parser('evaluateDetection',
    description='Evaluate detections in the open db with a ground truth db.')
  parser.set_defaults(func=evaluateDetection)
  parser.add_argument('--gt_db_file', required=True)
  parser.add_argument('--overlap_thresh', type=float, default=0.5)
  parser.add_argument('--where_object_gt', default='TRUE')

def evaluateDetection (c, args):

  # Attach the ground truth database.
  c.execute('ATTACH ? AS "gt"', (args.gt_db_file,))
  c.execute('SELECT DISTINCT(name) FROM gt.objects')
  names = [x for x, in c.fetchall()]
  aps = []  # Average precision per class.

  for name in names:

    c.execute('SELECT * FROM objects WHERE name=? ORDER BY score DESC', (name,))
    entries_det = c.fetchall()
    logging.info ('Total %d detected objects for class "%s"' % (len(entries_det), name))

    # go down dets and mark TPs and FPs
    tp = np.zeros(len(entries_det), dtype=float)
    fp = np.zeros(len(entries_det), dtype=float)
    ignored = np.zeros(len(entries_det), dtype=bool)  # detected of no interest

    # 'already_detected' used to penalize multiple detections of same GT box
    already_detected = set()

    # go through each detection
    for idet,entry_det in progressbar(enumerate(entries_det)):

      bbox_det = np.array(objectField(entry_det, 'bbox'), dtype=float)
      imagefile = objectField(entry_det, 'imagefile')
      score = objectField(entry_det, 'score')
      name = objectField(entry_det, 'name')

      # get all GT boxes from the same imagefile [of the same class]
      c.execute('SELECT * FROM gt.objects WHERE imagefile=? AND name=?', (imagefile,name))
      entries_gt = c.fetchall()
      objectids_gt = [objectField(entry, 'objectid') for entry in entries_gt]
      bboxes_gt = np.array([objectField(entry, 'bbox') for entry in entries_gt], dtype=float)

      # separately manage no GT boxes
      if bboxes_gt.size == 0:
        fp[idet] = 1.
        continue

      # intersection between bbox_det and all bboxes_gt.
      ixmin = np.maximum(bboxes_gt[:,0], bbox_det[0])
      iymin = np.maximum(bboxes_gt[:,1], bbox_det[1])
      ixmax = np.minimum(bboxes_gt[:,0]+bboxes_gt[:,2], bbox_det[0]+bbox_det[2])
      iymax = np.minimum(bboxes_gt[:,1]+bboxes_gt[:,3], bbox_det[1]+bbox_det[3])
      iw = np.maximum(ixmax - ixmin, 0.)
      ih = np.maximum(iymax - iymin, 0.)
      inters = iw * ih

      # union between bbox_det and all bboxes_gt.
      uni = bbox_det[2] * bbox_det[3] + bboxes_gt[:,2] * bboxes_gt[:,3] - inters

      # overlaps and get the best overlap.
      overlaps = inters / uni
      max_overlap = np.max(overlaps)
      objectid_gt = objectids_gt[np.argmax(overlaps)]

      # find which objects count towards TP and FN (should be detected).
      c.execute('SELECT * FROM gt.objects WHERE imagefile=? AND name=? AND %s' % 
        args.where_object_gt, (imagefile,name))
      entries_gt = c.fetchall()
      objectids_gt_of_interest = [objectField(entry, 'objectid') for entry in entries_gt]

      # if 1) large enough overlap and 2) this GT box was not detected before
      if max_overlap > args.overlap_thresh and not objectid_gt in already_detected:
        if objectid_gt in objectids_gt_of_interest:
          tp[idet] = 1.
        else:
          ignored[idet] = True
        already_detected.add(objectid_gt)
      else:
        fp[idet] = 1.

    # find the number of GT of interest
    c.execute('SELECT COUNT(1) FROM gt.objects WHERE %s AND name=?' % args.where_object_gt, (name,))
    n_gt = c.fetchone()[0]
    logging.info ('Total objects of interest: %d' % n_gt)

    # remove dets, neither TP or FP
    tp = tp[np.bitwise_not(ignored)]
    fp = fp[np.bitwise_not(ignored)]

    logging.info('ignored: %d, tp: %d, fp: %d, gt: %d' %
                 (np.count_nonzero(ignored),
                  np.count_nonzero(tp),
                  np.count_nonzero(fp),
                  n_gt))
    assert np.count_nonzero(tp) + np.count_nonzero(fp) + np.count_nonzero(ignored) == len(entries_det)

    # compute precision-recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(n_gt)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = _voc_ap(rec, prec)
    print ('Average precision for class "%s": %.3f' % (name, ap))
    aps.append(ap)
  print ('Mean average precision: %.3f' % np.array(aps).mean())



def fast_hist(a, b, n):
  k = (a >= 0) & (a < n)
  return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
  return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def calc_fw_iu(hist):
  pred_per_class = hist.sum(0)
  gt_per_class = hist.sum(1)
  return np.nansum(
      (gt_per_class * np.diag(hist)) / (pred_per_class + gt_per_class - np.diag(hist))) / gt_per_class.sum()

def calc_pixel_accuracy(hist):
  gt_per_class = hist.sum(1)
  return np.diag(hist).sum() / gt_per_class.sum()

def calc_mean_accuracy(hist):
  gt_per_class = hist.sum(1)
  acc_per_class = np.diag(hist) / gt_per_class
  return np.nanmean(acc_per_class)

def save_colorful_images(prediction, filename, palette, postfix='_color.png'):
  im = Image.fromarray(palette[prediction.squeeze()])
  im.save(filename[:-4] + postfix)

def label_mapping(input, mapping):
  output = np.copy(input)
  for ind in range(len(mapping)):
    output[input == mapping[ind][0]] = mapping[ind][1]
  return np.array(output, dtype=np.int64)

def plot_confusion_matrix(cm, classes,
    normalize=False, title='Confusion matrix', cmap=None):
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
    logging.info('Confusion matrix will be computed without normalization.')

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=90)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  plt.tight_layout()
  plt.ylabel('Ground truth')
  plt.xlabel('Predicted label')


def _label2classMapping(gt_mapping_dict, pred_mapping_dict):
  ''' Parse user-defined label mapping dictionaries. '''

  # "gt_mapping_dict" maps mask pixel-values to classes.
  labelmap_gt = literal_eval(gt_mapping_dict)
  labelmap_pr = literal_eval(pred_mapping_dict) if pred_mapping_dict else labelmap_gt
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
      raise ValueError('Class %s is in "pred_mapping_dict" but not in "gt_mapping_dict"')
    labelmap_pr_new[key] = class_names.index(labelmap_pr[key])
  labelmap_pr = labelmap_pr_new
  return labelmap_gt, labelmap_pr, class_names


def evaluateSegmentationIoUParser(subparsers):
  parser = subparsers.add_parser('evaluateSegmentationIoU',
    description='Evaluate mask segmentation w.r.t. a ground truth db.')
  parser.set_defaults(func=evaluateSegmentationIoU)
  parser.add_argument('--gt_db_file', required=True)
  parser.add_argument('--where_image', default='TRUE')
  parser.add_argument('--out_dir',
    help='If specified, output files with be written there.')
  parser.add_argument('--out_prefix', default='',
    help='Add to output filenames, use to keep predictions from different epochs in one dir.')
  parser.add_argument('--gt_mapping_dict', required=True,
    help='from ground truth maskfile to classes. E.g. "{0: \'background\', 255: \'car\'}"')
  parser.add_argument('--pred_mapping_dict', 
    help='the mapping from predicted masks to classes, if different from "gt_mapping_dict"')
  parser.add_argument('--class_to_record_iou',
    help='If specified, IoU for a class is recorded into the "score" field of the "images" table. '
    'If not specified, mean IoU is recorded. '
    'Should correspond to values of "gt_mapping_dict". E.g. "background".')
  parser.add_argument('--out_summary_file',
    help='Text file, where the summary is going to be appended as just one line of format: '
    'out_prefix \\t IoU_class1 \\t IoU_class2 \\t etc.')

def evaluateSegmentationIoU(c, args):
  import pandas as pd

  # Get corresponding maskfiles from predictions and ground truth.
  logging.info('Opening ground truth dataset: %s' % args.gt_db_file)
  c.execute('ATTACH ? AS "attached"', (args.gt_db_file,))
  c.execute('SELECT pr.imagefile,pr.maskfile,gt.maskfile FROM images pr INNER JOIN attached.images gt '
    'WHERE pr.imagefile=gt.imagefile AND pr.maskfile IS NOT NULL AND gt.maskfile IS NOT NULL AND %s '
    'ORDER BY pr.imagefile ASC' % args.where_image)
  entries = c.fetchall()
  logging.info ('Total %d images in both the open and the ground truth databases.' % len(entries))
  logging.debug (pformat(entries))

  imreader = MediaReader(rootdir=args.rootdir)

  labelmap_gt, labelmap_pr, class_names = _label2classMapping(
    args.gt_mapping_dict, args.pred_mapping_dict)

  if args.class_to_record_iou is not None and not args.class_to_record_iou in class_names:
    raise ValueError('class_to_record_iou=%s is not among values of gt_mapping_dict=%s' %
      (args.class_to_record_iou, args.gt_mapping_dict))

  hist_all = np.zeros((len(class_names), len(class_names)))

  for imagefile, maskfile_pr, maskfile_gt in progressbar(entries):

    # Load masks and bring them to comparable form.
    mask_gt = applyLabelMappingToMask(imreader.maskread(maskfile_gt), labelmap_gt)
    mask_pr = applyLabelMappingToMask(imreader.maskread(maskfile_pr), labelmap_pr)
    mask_pr = cv2.resize(mask_pr, (mask_gt.shape[1], mask_gt.shape[0]), interpolation=cv2.INTER_NEAREST)

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
    c.execute('UPDATE images SET score=? WHERE imagefile=?', (iou, imagefile))

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

#  print("---- result summary -----")
#  print(result_ser)

  if args.out_dir is not None:
    if not op.exists(args.out_dir):
      os.makedirs(args.out_dir)

    with open(op.join(args.out_dir, args.out_summary_file), 'a') as f:
        f.write(args.out_prefix + '\t' + '\t'.join(['%.2f' % x for x in result_df['IoU']]) + '\n')

    # Save confusion matrix
    fig = plt.figure()
    normalized_hist = hist.astype("float") / hist.sum(axis=1)[:, np.newaxis]

    plot_confusion_matrix(normalized_hist, classes=class_names, title='Confusion matrix')
    outfigfn = op.join(args.out_dir, "%sconf_mat.pdf" % args.out_prefix)
    fig.savefig(outfigfn, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    print("Confusion matrix was saved to %s" % outfigfn)

    outdffn = os.path.join(args.out_dir, "%seval_result_df.csv" % args.out_prefix)
    result_df.to_csv(outdffn)
    print('Info per class was saved at %s !' % outdffn)
    outserfn = os.path.join(args.out_dir, "%seval_result_ser.csv" % args.out_prefix)
    result_ser.to_csv(outserfn)
    print('Total result is saved at %s !' % outserfn)



def getPrecRecall(tp, fp, tn, fn):
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
  ROC = ROC[np.bitwise_and(ROC[:,0] != -1, ROC[:,1] != -1), :]
  ROC = np.vstack(( ROC, np.array([0, ROC[-1,1]]) ))
  area = -np.trapz(x=ROC[:,0], y=ROC[:,1])
  return ROC, area

def evaluateBinarySegmentationParser(subparsers):
  parser = subparsers.add_parser('evaluateBinarySegmentation',
      description='Evaluate mask segmentation ROC curve w.r.t. a ground truth db. '
      'Ground truth values must be 0 for background, 255 for foreground, and the rest for dontcare.'
      'Predicted mask must be grayscale in [0,255], with brightness meaning probability of foreground.')
  parser.set_defaults(func=evaluateBinarySegmentation)
  parser.add_argument('--gt_db_file', required=True)
  parser.add_argument('--where_image', default='TRUE')
  parser.add_argument('--out_dir',
      help='If specified, result files with be written there.')
  parser.add_argument('--out_prefix', default='',
      help='Add to filenames, use to keep predictions from different epochs in one dir.')
  parser.add_argument('--display_images_roc', action='store_true')

def evaluateBinarySegmentation(c, args):
  import pandas as pd

  # Get corresponding maskfiles from predictions and ground truth.
  c.execute('ATTACH ? AS "attached"', (args.gt_db_file,))
  c.execute('SELECT pr.imagefile,pr.maskfile,gt.maskfile FROM images pr INNER JOIN attached.images gt '
    'WHERE pr.imagefile=gt.imagefile AND pr.maskfile IS NOT NULL AND gt.maskfile IS NOT NULL AND %s '
    'ORDER BY pr.imagefile ASC' % args.where_image)
  entries = c.fetchall()
  logging.info ('Total %d images in both the open and the ground truth databases.' % len(entries))
  logging.debug (pformat(entries))

  imreader = MediaReader(rootdir=args.rootdir)

  TPs = np.zeros((256,), dtype=int)
  TNs = np.zeros((256,), dtype=int)
  FPs = np.zeros((256,), dtype=int)
  FNs = np.zeros((256,), dtype=int)

  if args.display_images_roc:
    fig = plt.figure()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

  for imagefile, maskfile_pr, maskfile_gt in progressbar(entries):

    # Load masks and bring them to comparable form.
    mask_gt = imreader.maskread(maskfile_gt)
    mask_pr = imreader.maskread(maskfile_pr)
    mask_pr = cv2.resize(mask_pr, (mask_gt.shape[1], mask_gt.shape[0]), cv2.INTER_NEAREST)

    # Some printputs.
    gt_pos = np.count_nonzero(mask_gt == 255)
    gt_neg = np.count_nonzero(mask_gt == 0)
    gt_other = mask_gt.size - gt_pos - gt_neg
    logging.debug('GT: positive: %d, negative: %d, others: %d.' % (gt_pos, gt_neg, gt_other))

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

      TP = np.zeros((256,), dtype=int)
      TN = np.zeros((256,), dtype=int)
      FP = np.zeros((256,), dtype=int)
      FN = np.zeros((256,), dtype=int)
      for val in range(256):
        tp = torch.nonzero(torch.mul(mask_pr > val, mask_gt == 255)).size()[0]
        fp = torch.nonzero(torch.mul(mask_pr > val, mask_gt != 255)).size()[0]
        fn = torch.nonzero(torch.mul(mask_pr <= val, mask_gt == 255)).size()[0]
        tn = torch.nonzero(torch.mul(mask_pr <= val, mask_gt != 255)).size()[0]
        TP[val] = tp
        FP[val] = fp
        TN[val] = tn
        FN[val] = fn
        TPs[val] += tp
        FPs[val] += fp
        TNs[val] += tn
        FNs[val] += fn
      ROC, area = getPrecRecall(TP, FP, TN, FN)
      logging.info('%s\t%.2f' % (op.basename(imagefile), area * 100.))

    except ImportError:
      # TODO: write the same without torch, on CPU
      raise NotImplementedError('Still need to write a non-torch implementation.')
    
    if args.display_images_roc:
      plt.plot(ROC[:, 0], ROC[:, 1], 'go-', linewidth=2, markersize=4)
      plt.pause(0.05)
      fig.show()

  # Accumulate into Precision-Recall curve.
  ROC, area = getPrecRecall(TPs, FPs, TNs, FNs)
  print("Average across image area under the Precision-Recall curve, perc: %.2f" % (area * 100.))

  if args.out_dir is not None:
    if not op.exists(args.out_dir):
      os.makedirs(args.out_dir)
    fig = plt.figure()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(ROC[:, 0], ROC[:, 1], 'bo-', linewidth=2, markersize=6)
    out_plot_path = op.join(args.out_dir, '%srecall-prec.png' % args.out_prefix)
    fig.savefig(out_plot_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
