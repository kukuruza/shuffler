#! /usr/bin/env python3
import sys, os, os.path as op
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from argparse import ArgumentParser
import progressbar
import logging
import imageio

from lib.backend import backendMedia


def erodeMask_parser():
    parser = ArgumentParser(
        'Erode one color in a mask, in favor or another color.')
    parser.add_argument('-i', '--in_mask_path', required=True)
    parser.add_argument('-o', '--out_mask_path', required=True)
    parser.add_argument('--dryrun',
                        action='store_true',
                        help='Do not write anything.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--color_eroding', type=int, required=True)
    parser.add_argument('--color_instead', type=int, required=True)
    parser.add_argument('--depth', type=int, default=1)
    return parser


def erodeMask(args):
    imreader = imageio.get_reader(args.in_mask_path)
    if not args.dryrun:
        imwriter = backendMedia.VideoWriter(vmaskfile=args.out_mask_path,
                                            overwrite=args.overwrite)

    for i in progressbar.progressbar(range(imreader.get_length())):
        mask = imreader.get_data(i)
        roi = mask == args.color_eroding
        roieroded = cv2.erode(roi.astype(np.uint8),
                              kernel=np.ones((3, 3)),
                              iterations=args.depth,
                              borderValue=0).astype(bool)
        maskbefore = mask.copy()
        mask[roi] = args.color_instead
        mask[roieroded] = args.color_eroding
        if not args.dryrun:
            imwriter.maskwrite(mask)

    if not args.dryrun:
        imwriter.close()


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
