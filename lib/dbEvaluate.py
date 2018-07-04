import os, os.path as op
import sys
import logging
import sqlite3
import numpy as np
import cv2
import torch
from progressbar import progressbar
import simplejson as json
import matplotlib
from matplotlib import pyplot as plt
from pprint import pformat
from backendDb import carField, loadToMemory
from backendImages import maskread, validateMask
from utilities import loadLabelmap



def add_parsers(subparsers):
  evaluateSegmentationIoUParser(subparsers)
  evaluateSegmentationROCParser(subparsers)


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

def evaluateSegmentationIoUParser(subparsers):
  parser = subparsers.add_parser('evaluateSegmentationIoU',
    description='Evaluate mask segmentation w.r.t. a ground truth db.')
  parser.set_defaults(func=evaluateSegmentationIoU)
  parser.add_argument('--gt_db_file', required=True)
  parser.add_argument('--image_constraint', default='1')
  parser.add_argument('--out_dir',
      help='If specified, result files with be written there.')
  parser.add_argument('--out_prefix', default='',
      help='Add to filenames, use to keep predictions from different epochs in one dir.')
  parser.add_argument('--gt_labelmap_path')
  parser.add_argument('--pred_labelmap_path')
  parser.add_argument('--debug_display', action='store_true')
  parser.add_argument('--debug_winwidth', type=int, default=1000)

def evaluateSegmentationIoU(c, args):
  logging.info ('==== evaluateSegmentationIoU ====')
  import pandas as pd

  s = ('SELECT imagefile, maskfile FROM images '
       'WHERE %s ORDER BY imagefile ASC' % args.image_constraint)
  logging.debug(s)

  c.execute(s)
  entries_pred = c.fetchall()
  logging.info ('Total %d images in predicted.' % len(entries_pred))

  conn_gt = loadToMemory(args.gt_db_file)
  c_gt = conn_gt.cursor()
  c_gt.execute(s)
  entries_gt = c_gt.fetchall()
  logging.info ('Total %d images in gt.' % len(entries_gt))
  conn_gt.close()

  # Map from label on the image to the target label space.
  labelmap_pred = loadLabelmap(args.pred_labelmap_path) if args.pred_labelmap_path else {}
  labelmap_gt = loadLabelmap(args.gt_labelmap_path) if args.gt_labelmap_path else {}
  def relabelGrayMask(mask, labelmap):
    ''' Paint mask colors with other colors, grayscale to grayscale. '''
    out = np.zeros(shape=mask.shape, dtype=np.uint8)
    for key in labelmap:
      out[mask == key] = labelmap[key]['to']
    return out
  def getRelevantPixels(mask, labelmap):
    ''' Return pixels that are used for evaluation '''
    care_labels = [labelmap[key]['to'] for key in labelmap if 'class' in labelmap[key]]
    care_pixels = np.zeros(mask.shape, dtype=bool)
    for care_label in care_labels:
      care_pixels[mask == care_label] = True
    return care_pixels

  class_names = [labelmap_gt[key]['class'] for key in labelmap_gt if 'class' in labelmap_gt[key]]
  num_relevant_classes = len(class_names)
  hist = np.zeros((num_relevant_classes, num_relevant_classes))

  for entry_pred, entry_gt in progressbar(zip(entries_pred, entries_gt)):
    imagefile_pred, maskfile_pred = entry_pred
    imagefile_gt, maskfile_gt = entry_gt
    if op.basename(imagefile_pred) != op.basename(imagefile_gt):
      raise ValueError('Imagefile entries for pred and gt must be the same: %s vs %s.' %
          (imagefile_pred, imagefile_gt))
    if maskfile_pred is None:
      raise ValueError('Mask is None for image %s' % imagefile_pred)
    if maskfile_gt is None:
      raise ValueError('Mask is None for image %s' % imagefile_gt)

    # Load masks and bring them to comparable form.
    mask_gt = relabelGrayMask(validateMask(maskread(maskfile_gt)), labelmap_gt)
    mask_pred = relabelGrayMask(validateMask(maskread(maskfile_pred)), labelmap_pred)
    mask_pred = cv2.resize(mask_pred, (mask_gt.shape[1], mask_gt.shape[0]), cv2.INTER_NEAREST)
    if args.debug_display:
      scale = float(args.debug_winwidth) / mask_pred.shape[1]
      cv2.imshow('debug_display', cv2.resize((np.hstack((mask_gt, mask_pred)) != 0).astype(np.uint8), dsize=(0,0), fx=scale, fy=scale) * 255)
      if cv2.waitKey(-1) == 27:
        args.debug_display = False

    # Use only relevant pixels (not the 'dontcare' class.)
    relevant = getRelevantPixels(mask_gt, labelmap_gt)
    mask_gt = mask_gt[relevant].flatten()
    mask_pred = mask_pred[relevant].flatten()
    assert np.all(mask_gt < num_relevant_classes), np.max(num_relevant_classes)
    assert np.all(mask_pred < num_relevant_classes), np.max(num_relevant_classes)

    # Evaluate one image pair. 
    hist += fast_hist(mask_gt, mask_pred, 2)

  # Get label distribution.
  pred_per_class = hist.sum(0)
  gt_per_class = hist.sum(1)

  # Remove the classes that have >0 pixels in 'pred' and in 'gt'.
  # so that we don't experience zero division further on.
  used_class_id_list = np.where(gt_per_class != 0)[0]
  hist = hist[used_class_id_list][:, used_class_id_list]
  class_list = np.array(class_names)[used_class_id_list]

  iou_list = per_class_iu(hist)
  fwIoU = calc_fw_iu(hist)
  pixAcc = calc_pixel_accuracy(hist)
  mAcc = calc_mean_accuracy(hist)

  result_df = pd.DataFrame({
      'class': class_names,
      'IoU': iou_list,
      "pred_distribution": pred_per_class[used_class_id_list],
      "gt_distribution": gt_per_class[used_class_id_list],
  })
  result_df["IoU"] = result_df["IoU"] * 100  # change to percent ratio

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

  print("---- total result -----")
  print(result_ser)

  if args.out_dir is not None:
    if not op.exists(args.out_dir):
      os.makedirs(args.out_dir)

    # Save confusion matrix
    fig = plt.figure()
    normalized_hist = hist.astype("float") / hist.sum(axis=1)[:, np.newaxis]

    plot_confusion_matrix(normalized_hist, classes=class_list, title='Confusion matrix')
    outfigfn = os.path.join(args.out_dir, "%sconf_mat.pdf" % args.out_prefix)
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

def evaluateSegmentationROCParser(subparsers):
  parser = subparsers.add_parser('evaluateSegmentationROC',
      description='Evaluate mask segmentation ROC curve w.r.t. a ground truth db. '
      'GT mask must be relabelled into 0 and 1, and "dontcare" using "gt_labelmap_file". '
      'Pred mask must be grayscale in [0,255], with brightness meaning probability of class 1.')
  parser.set_defaults(func=evaluateSegmentationROC)
  parser.add_argument('--gt_db_file', required=True)
  parser.add_argument('--image_constraint', default='1')
  parser.add_argument('--out_dir',
      help='If specified, result files with be written there.')
  parser.add_argument('--out_prefix', default='',
      help='Add to filenames, use to keep predictions from different epochs in one dir.')
  parser.add_argument('--display_images_roc', action='store_true')
  parser.add_argument('--debug_display', action='store_true')
  parser.add_argument('--debug_winwidth', type=int, default=1000)

def evaluateSegmentationROC(c, args):
  logging.info ('==== evaluateSegmentationROC ====')
  import pandas as pd

  s = ('SELECT imagefile, maskfile FROM images '
       'WHERE %s ORDER BY imagefile ASC' % args.image_constraint)
  logging.debug(s)

  c.execute(s)
  entries_pred = c.fetchall()
  logging.info ('Total %d images in predicted.' % len(entries_pred))

  conn_gt = loadToMemory(args.gt_db_file)
  c_gt = conn_gt.cursor()
  c_gt.execute(s)
  entries_gt = c_gt.fetchall()
  logging.info ('Total %d images in gt.' % len(entries_gt))
  conn_gt.close()

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

  for entry_pred, entry_gt in progressbar(zip(entries_pred, entries_gt)):
    imagefile_pred, maskfile_pred = entry_pred
    imagefile_gt, maskfile_gt = entry_gt
    if op.basename(imagefile_pred) != op.basename(imagefile_gt):
      raise ValueError('Imagefile entries for pred and gt must be the same: %s vs %s.' %
          (imagefile_pred, imagefile_gt))
    if maskfile_pred is None:
      raise ValueError('Mask is None for image %s' % imagefile_pred)
    if maskfile_gt is None:
      raise ValueError('Mask is None for image %s' % imagefile_gt)

    # Load masks and bring them to comparable form.
    mask_gt = validateMask(maskread(maskfile_gt))
    mask_pred = validateMask(maskread(maskfile_pred))
    mask_pred = cv2.resize(mask_pred, (mask_gt.shape[1], mask_gt.shape[0]), cv2.INTER_NEAREST)
    if args.debug_display:
      if cv2.waitKey(-1) == 27:
        args.debug_display = False
      scale = float(args.debug_winwidth) / mask_pred.shape[1]
      cv2.imshow('debug_display', cv2.resize(np.hstack((mask_gt, mask_pred)), dsize=(0,0), fx=scale, fy=scale))

    # Some printputs.
    gt_pos = np.count_nonzero(mask_gt == 255)
    gt_neg = np.count_nonzero(mask_gt == 0)
    gt_other = mask_gt.size - gt_pos - gt_neg
    logging.debug('GT: positive: %d, negative: %d, others: %d.' % (gt_pos, gt_neg, gt_other))

    # Use only relevant pixels (not the 'dontcare' class.)
    relevant = np.bitwise_or(mask_gt == 0, mask_gt == 255)
    mask_gt = mask_gt[relevant].flatten()
    mask_pred = mask_pred[relevant].flatten()
    mask_gt = torch.Tensor(mask_gt).cuda()
    mask_pred = torch.Tensor(mask_pred).cuda()

    TP = np.zeros((256,), dtype=int)
    TN = np.zeros((256,), dtype=int)
    FP = np.zeros((256,), dtype=int)
    FN = np.zeros((256,), dtype=int)
    for val in range(256):
      tp = torch.nonzero(torch.mul(mask_pred > val, mask_gt == 255)).size()[0]
      fp = torch.nonzero(torch.mul(mask_pred > val, mask_gt != 255)).size()[0]
      fn = torch.nonzero(torch.mul(mask_pred <= val, mask_gt == 255)).size()[0]
      tn = torch.nonzero(torch.mul(mask_pred <= val, mask_gt != 255)).size()[0]
      TP[val] = tp
      FP[val] = fp
      TN[val] = tn
      FN[val] = fn
      TPs[val] += tp
      FPs[val] += fp
      TNs[val] += tn
      FNs[val] += fn
    ROC, area = getPrecRecall(TP, FP, TN, FN)
    logging.info('%s\t%.2f' % (op.basename(imagefile_gt), area * 100.))
    
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
