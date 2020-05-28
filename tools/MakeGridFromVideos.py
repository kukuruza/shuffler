#! /usr/bin/env python3
import sys, os, os.path as op
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from glob import glob
from argparse import ArgumentParser
import progressbar
import logging
import imageio

from lib.backend import backendMedia


def MakeGridFromVideos_parser():
    parser = ArgumentParser(
        'Take a list of video files, and make a new one, '
        'where each frame is a grid of frames from the input videos.')
    parser.add_argument('-i', '--in_video_paths', nargs='+', required=True)
    parser.add_argument('-o', '--out_video_path', required=True)
    parser.add_argument(
        '--gridY',
        type=int,
        help='if specified, use the grid of "gridY" cells wide. Infer gridX.')
    parser.add_argument('--dryrun',
                        action='store_true',
                        help='Do not write anything.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--imwidth', type=int, required=True)
    return parser


def MakeGridFromVideos(args):
    num_videos = len(args.in_video_paths)
    gridY = int(sqrt(num_videos)) if args.gridY is None else args.gridY
    gridX = (num_videos - 1) // gridY + 1
    logging.info('Will use grid %d x %d' % (gridY, gridX))

    if not args.dryrun:
        writer = backendMedia.VideoWriter(vmaskfile=args.out_video_path,
                                          overwrite=args.overwrite,
                                          fps=args.fps)

    def processImage(image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(
            image.shape) == 2 else image
        scale = float(args.imwidth) / image.shape[1]
        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
        return image

    handles = [imageio.get_reader(path) for path in args.in_video_paths]
    #for handle, path in zip(handles, args.in_video_paths):
    #    logging.info('%d frames in video %s.' % (handle.get_length(), path))
    num_frames = 26 #min([handle.get_length() for handle in handles])

    print ('gridX, gridY', gridX, gridY)

    for i in progressbar.progressbar(range(num_frames)):
        images = [handle.get_data(i) for handle in handles]

        images = [processImage(image) for image in images]
        height, width = images[0].shape[0:2]

        # Lazy initialization.
        if 'grid' not in locals():
            grid = np.zeros((height * gridY, width * gridX, 3), dtype=np.uint8)

        for gridid, image in enumerate(images):
            grid[height * (gridid // gridX):height * (gridid // gridX + 1),
                 width * (gridid % gridX):width *
                 (gridid % gridX + 1), :] = image.copy()

        if not args.dryrun:
            writer.maskwrite(grid)

    for handle in handles:
        handle.close()


if __name__ == '__main__':
    parser = MakeGridFromVideos_parser()
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    args = parser.parse_args()
    print(args)

    progressbar.streams.wrap_stderr()
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    MakeGridFromVideos(args)
