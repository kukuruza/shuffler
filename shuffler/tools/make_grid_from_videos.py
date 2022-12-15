#! /usr/bin/env python3
import numpy as np
import cv2
import argparse
import progressbar
import logging
import imageio

from shuffler.backend import backend_media


def getParser():
    parser = argparse.ArgumentParser(
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
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    return parser


def MakeGridFromVideos(args):
    num_videos = len(args.in_video_paths)
    gridY = int(np.sqrt(num_videos)) if args.gridY is None else args.gridY
    gridX = (num_videos - 1) // gridY + 1
    logging.info('Will use [gridY x gridX] = [%d x %d].', gridY, gridX)

    if not args.dryrun:
        writer = backend_media.VideoWriter(vmaskfile=args.out_video_path,
                                           overwrite=args.overwrite,
                                           fps=args.fps)

    def processImage(image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(
            image.shape) == 2 else image
        scale = float(args.imwidth) / image.shape[1]
        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
        return image

    handles = [imageio.get_reader(path) for path in args.in_video_paths]
    grid = None

    for frames in progressbar.progressbar(zip(*handles)):
        frames = [processImage(frame) for frame in frames]
        height, width = frames[0].shape[0:2]

        # Lazy initialization.
        if grid is None:
            grid = np.zeros((height * gridY, width * gridX, 3), dtype=np.uint8)

        for gridid, image in enumerate(frames):
            grid[height * (gridid // gridX):height * (gridid // gridX + 1),
                 width * (gridid % gridX):width *
                 (gridid % gridX + 1), :] = image.copy()

        if not args.dryrun:
            writer.maskwrite(grid)

    for handle in handles:
        handle.close()


if __name__ == '__main__':
    args = getParser().parse_args()

    progressbar.streams.wrap_stderr()
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    MakeGridFromVideos(args)
