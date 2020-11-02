#! /usr/bin/env python3
import sys, os, os.path as op
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from argparse import ArgumentParser
import progressbar
import logging
import imageio
from glob import glob


def erodeMask_parser():
    parser = ArgumentParser(
        'Erode one color in a mask, in favor or another color.')
    parser.add_argument('-i', '--in_mask_pattern', required=True)
    parser.add_argument('-o', '--out_mask_dir', required=True)
    parser.add_argument('--dryrun',
                        action='store_true',
                        help='Do not write anything.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--color_eroding', type=int, required=True)
    parser.add_argument('--color_instead', type=int, required=True)
    parser.add_argument('--depth', type=int, default=1)
    return parser


def erodeMask(args):
    in_mask_paths = glob(args.in_mask_pattern)

    if not args.dryrun and not op.exists(args.out_mask_dir):
        os.makedirs(args.out_mask_dir)

    for in_mask_path in progressbar.progressbar(in_mask_paths):
        mask = imageio.imread(in_mask_path)
        roi = mask == args.color_eroding
        roieroded = cv2.erode(roi.astype(np.uint8),
                              kernel=np.ones((3, 3)),
                              iterations=args.depth,
                              borderValue=0).astype(bool)
        maskbefore = mask.copy()
        mask[roi] = args.color_instead
        mask[roieroded] = args.color_eroding
        out_mask_path = op.join(args.out_mask_dir, op.basename(in_mask_path))
        if not args.dryrun:
            if op.exists(out_mask_path) and not args.overwrite:
                raise Exception('File %s exists' % out_mask_path)
            imageio.imwrite(out_mask_path, mask)


if __name__ == '__main__':
    parser = erodeMask_parser()
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    args = parser.parse_args()

    progressbar.streams.wrap_stderr()
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    erodeMask(args)
