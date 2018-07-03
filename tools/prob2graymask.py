import os.path as op
import torch
import numpy as np
import cv2
from glob import glob
from argparse import ArgumentParser
import progressbar
import logging

def prob2graymask_parser():
  parser = ArgumentParser('Convert numpy or pytorch data files '
      'with probabilities to grayscale images for one class.')
  parser.add_argument('-i', '--in_path_pattern', required=True,
      help='E.g. "mydir/*.dat" or "mydir/*.npy". Remember to mask * with quotes.')
  parser.add_argument('-o', '--out_path_pattern', required=True,
      help='E.g. "mydir/label0-*.png"')
  parser.add_argument('--dryrun', action='store_true', help='Do not write anything.')
  parser.add_argument('--softmax', action='store_true', help='Use softmax on top of prbabilties.')
  parser.add_argument('--labelid', type=int, required=True,
      help='Probabilites of multiple ')
  return parser


def prob2graymask(args):

  out_dir = op.dirname(args.out_path_pattern)
  if not op.exists(op.dirname(out_dir)) and not args.dryrun:
    os.makedirs(op.dirname(out_dir))

  in_data_paths = sorted(glob(args.in_path_pattern))
  logging.info('Will convert %d files.' % len(in_data_paths))

  for in_data_path in progressbar.progressbar(in_data_paths):

    # Load file.
    ext = op.splitext(in_data_path)[1]
    if ext == '.npy':
      logging.debug('Processing %s as numpy file.' % in_data_path)
      prob = np.load(in_data_path)
    elif ext == '.dat':
      logging.debug('Processing %s as pytorch file.' % in_data_path)
      prob = torch.load(in_data_path).numpy()
    else:
      raise Exception('File must be either .dat for pytorch or .npy for numpy, but it is "%s"' % ext)

    # Apply softmax if needed.
    if args.softmax:
      prob = np.exp(prob) / np.tile(np.sum(np.exp(prob), axis=0), reps=[prob.shape[0],1,1])

    # Slice the probability of a particular label.
    if len(prob.shape) != 3:
      raise ValueError('Expect prob. shape of shape NxHxW, not %s' % prob.shape)
    if args.labelid > prob.shape[0] or args.labelid < 0:
      raise ValueError('Expect labelid be in range of prob.shape[0], which is 0-%d' % prob.shape[0])
    prob = prob[args.labelid]

    # Scale from 0-1 float to 0-255 uint8.
    if np.any(prob < 0) or np.any(prob > 1):
      raise ValueError('Expect probabilities in [0,1] but they are in [%.3f, %.3f]' % (prob.min(), prob.max()))
    prob = (prob * 255.).astype(np.uint8)
    logging.debug('Probabilities are in [%.3f, %.3f]' % (prob.min(), prob.max()))

    # Save as image.
    # Two split ext to get read of the extra .jpg suffix
    out_name = op.splitext(op.splitext(op.basename(in_data_path))[0])[0]
    if args.out_path_pattern.count('*') != 1:
      raise Exception('out_path_pattern should have exactly one "*" character.')
    out_path = args.out_path_pattern.replace('*', out_name)
    logging.debug('Will write the mask as %s' % out_path)
    assert len(prob.shape) == 2, prob.shape
    if not args.dryrun:
      cv2.imwrite(out_path, prob)


if __name__ == '__main__':
  parser = prob2graymask_parser()
  parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
    help='Log debug (10), info (20), warning (30), error (40).')
  args = parser.parse_args()

  progressbar.streams.wrap_stderr()
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  prob2graymask(args)
  