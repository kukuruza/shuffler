import os, os.path as op
import sys
import logging
import sqlite3
import numpy as np
from progressbar import progressbar
import simplejson as json
from backendDb import carField, loadToMemory
from backendImages import maskread, validateMask
from utilities import relabelGrayMask



def add_parsers(subparsers):
  evaluateSegmentationParser(subparsers)


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
  import matplotlib
  matplotlib.use('Agg')
  from matplotlib import pyplot as plt

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

def evaluateSegmentationParser(subparsers):
  parser = subparsers.add_parser('evaluateSegmentation',
    description='Evaluate mask segmentation w.r.t. a ground truth db.')
  parser.set_defaults(func=evaluateSegmentation)
  parser.add_argument('--gt_db_file', required=True)
  parser.add_argument('--image_constraint', default='1')
  parser.add_argument('--out_dir',
      help='If specified, result files with be written there.')
  parser.add_argument('--out_prefix', default='',
      help='Add to filenames, use to keep predictions from different epochs in one dir.')
  parser.add_argument('--gt_labelmap_file', required=True)

def evaluateSegmentation(c, args):
  logging.info ('==== evaluateSegmentation ====')
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

  hist = np.zeros((2, 2))

  if not op.exists(args.pred_labelmap_file):
    raise Exception('Pred labelmap file does not exist: %s' % args.pred_labelmap_file)
  with open(args.pred_labelmap_file) as f:
    pred_labelmap = json.load(f)['label']

  if not op.exists(args.gt_labelmap_file):
    raise Exception('GT labelmap file does not exist: %s' % args.gt_labelmap_file)
  with open(args.gt_labelmap_file) as f:
    gt_labelmap = json.load(f)['label']

  for entry_pred, entry_gt in progressbar(zip(entries_pred, entries_gt)):
    imagefile_pred, maskfile_pred = entry_pred
    imagefile_gt, maskfile_gt = entry_gt
    if imagefile_pred != imagefile_gt:
      raise ValueError('Imagefile entries for pred and gt must be the same: %s vs %s.' %
          (imagefile_pred, imagefile_gt))

    mask_pred = relabelGrayMask(validateMask(maskread(maskfile_pred)), pred_labelmap)
    mask_gt = relabelGrayMask(validateMask(maskread(maskfile_gt)), gt_labelmap)

    hist += fast_hist(mask_gt.flatten(), mask_pred.flatten(), 2)

  # Get label distribution
  pred_per_class = hist.sum(0)
  gt_per_class = hist.sum(1)

  used_class_id_list = np.where(gt_per_class != 0)[0]
  hist = hist[used_class_id_list][:, used_class_id_list]  # Extract only GT existing (more than 1) classes

  class_list = np.array(['background', 'car'])[used_class_id_list]

  iou_list = per_class_iu(hist)
  fwIoU = calc_fw_iu(hist)
  pixAcc = calc_pixel_accuracy(hist)
  mAcc = calc_mean_accuracy(hist)

  result_df = pd.DataFrame({
      'class': ['background', 'car'],
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
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

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

