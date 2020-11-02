#! /usr/bin/env python3
import sys, os, os.path as op
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import argparse
import progressbar
import logging
import imageio

from lib.backend import backendMedia


def subtractMask_parser():
    parser = argparse.ArgumentParser('Take a difference between masks')
    parser.add_argument('-1', '--in_mask_path1', required=True)
    parser.add_argument('-2', '--in_mask_path2', required=True)
    parser.add_argument('-o', '--out_mask_path', required=True)
    parser.add_argument('--dryrun',
                        action='store_true',
                        help='Do not write anything.')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def subtractMask(args):
    imreader1 = imageio.get_reader(args.in_mask_path1)
    imreader2 = imageio.get_reader(args.in_mask_path2)
    logging.info('%d frames in video %s.' %
                 (imreader1.get_length(), args.in_mask_path1))
    logging.info('%d frames in video %s.' %
                 (imreader2.get_length(), args.in_mask_path2))
    if not args.dryrun:
        imwriter = backendMedia.VideoWriter(vmaskfile=args.out_mask_path,
                                            overwrite=args.overwrite)

    num_frames = min([imreader1.get_length(), imreader2.get_length()])

    for i in progressbar.progressbar(range(num_frames)):
        mask1 = imreader1.get_data(i)
        mask2 = imreader2.get_data(i)
        maskdiff = (mask1.astype(int) - mask2.astype(int)).astype(np.uint8)

        if not args.dryrun:
            imwriter.maskwrite(maskdiff)

    if not args.dryrun:
        imwriter.close()


if __name__ == '__main__':
    parser = subtractMask_parser()
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

    subtractMask(args)
