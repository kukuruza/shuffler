import os, sys, os.path as op
import logging
from glob import glob
from argparse import ArgumentParser
import progressbar
import cv2


def images2video_parser():
  parser = ArgumentParser('Convert images to video.')
  parser.add_argument('-i', '--in_images_dir', required=True)
  parser.add_argument('-o', '--out_video_path', required=True)
  parser.add_argument('--ext', default='jpg', help='Extension of images.')
  parser.add_argument('--overwrite', action='store_true', help='Overwrite video if it exists.')
  parser.add_argument('--fps', type=int, default=2, help='Video FPS rate.')
  parser.add_argument('--fourcc', type=int, default=1196444237, help='Opencv code for the codec.')
  parser.add_argument('--dryrun', action='store_true', help='Do not write anything.')
  return parser


def images2video (args):

  # check if dir exists
  if not op.exists(op.dirname(args.out_video_path)):
    os.makedirs(op.dirname(args.out_video_path))
  elif op.exists(args.out_video_path):
    if args.overwrite:
      os.remove(args.out_video_path)
    else:
      raise IOError('images2video: video file already exists: %s. Set overwrite=True.' % args.out_video_path)

  if not op.exists(args.in_images_dir):
    raise IOError('images2video: image directory does not exist: %s' % args.in_images_dir)

  video = None

  in_image_paths = sorted(glob(op.join(args.in_images_dir, '*.%s' % args.ext)))
  logging.info('images2video: will write %d frames.' % len(in_image_paths))

  for in_image_path in progressbar.progressbar(in_image_paths):
    logging.debug('images2video: processing %s' % in_image_path)
    image = cv2.imread(in_image_path)
    if image is None:
      raise IOError('images2video: error reading %s' % in_image_path)

    # Lazy video initialization.
    if video is None:
      first_path = in_image_path
      (height, width) = image.shape[0:2]
      iscolor = True if len(image.shape) == 3 else 2
      video = cv2.VideoWriter(args.out_video_path, args.fourcc, args.fps, (width, height), isColor=iscolor)
    # Checking frame size and color.
    else:
      if (len(image.shape) == 3) != iscolor:
        raise ValueError('images2video: images must be either grayscale or color: %s vs %s'
            (first_path, in_image_path))
      if not image.shape[0:2] == (height, width):
        raise ValueError('images2video: shapes of images mismatch: %s vs %s: %dx%d vs %dx%d.'
            (first_path, in_image_path, height, width, image.shape[0], image.shape[1]))

    if not args.dryrun:
      video.write(image)


if __name__ == '__main__':
  parser = images2video_parser()
  parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
    help='Log debug (10), info (20), warning (30), error (40).')
  args = parser.parse_args()

  progressbar.streams.wrap_stderr()
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  images2video(args)
  