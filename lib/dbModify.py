import os, sys, os.path as op
import numpy as np
import cv2
import logging
from glob import glob
from pprint import pformat
from backendImages import getVideoLength


def add_parsers(subparsers):
  importVideoParser(subparsers)
  importPicturesParser(subparsers)


def importVideoParser(subparsers):
  parser = subparsers.add_parser('importVideo',
      description='Import frames from a video into the database.')
  parser.add_argument('--frame_video_path', required=True)
  parser.add_argument('--mask_video_path')
  parser.add_argument('--src', help='Where the video came from.')
  parser.set_defaults(func=importVideo)

def importVideo (c, args):
  logging.info ('==== importVideo ====')

  # Check video paths.
  if not op.exists(args.frame_video_path):
    raise IOError('frame video does not exist: %s' % args.frame_video_path)
  if args.mask_video_path is not None and not op.exists(args.mask_video_path):
    raise IOError('mask video does not exist: %s' % args.mask_video_path)

  frame_video = cv2.VideoCapture(args.frame_video_path)
  ret, frame = frame_video.read()
  length = getVideoLength(frame_video)
  logging.info('Video has %d frames' % length)
  if not ret:
    raise IOError('Frame video %s is empty' % args.frame_video_path)

  # Check how mask agrees with images.
  if args.mask_video_path is not None:
    mask_video = cv2.VideoCapture(args.mask_video_path)
    ret, mask = mask_video.read()
    if not ret:
      raise IOError('Mask video %s is empty' % args.mask_video_path)
    if not frame.shape[0] == mask.shape[0] or frame.shape[1] == mask.shape[1]:
      raise ValueError('Frame size %s and mask size %s mismatch' % (frame.shape[1], mask.shape[1]))
    mask_length = getVideoLength(mask_video)
    if length != mask_length:
      raise ValueError('Frame length %d mask video length %d mismatch' % (length, mask_length))

  # Get the paths.
  relpath = os.getenv('HOME')
  frame_video_rel_path = op.relpath(op.abspath(args.frame_video_path), relpath)
  if args.mask_video_path is not None:
    mask_video_rel_path = op.relpath(op.abspath(args.frame_video_path), relpath)

  # Write to db.
  for iframe in range(length):
    h, w = frame.shape[0:2]
    imagefile = op.join(frame_video_rel_path, '%06d' % iframe)
    maskfile = op.join(mask_video_rel_path, '%06d' % iframe) if args.mask_video_path else None
    s = 'INSERT INTO images(imagefile,maskfile,src,width,height) VALUES (?,?,?,?,?)'
    c.execute(s, (imagefile, maskfile, args.src, w, h))


def importPicturesParser(subparsers):
  parser = subparsers.add_parser('importPictures',
      description='Import picture files into the database.'
      'File names (except extentions) for images and masks must match.')
  parser.add_argument('--image_pattern', required=True,
      help='remember to put quotes around "*", e.g. "my/path/images-*.jpg"')
  parser.add_argument('--mask_pattern', default='/dummy',
      help='remember to put quotes around "*", e.g. "my/path/masks-*.jpg"')
  parser.add_argument('--src',
      help='Where the images came from.')
  parser.set_defaults(func=importPictures)

def importPictures (c, args):
  logging.info ('==== importPictures ====')

  # Collect a list of paths.
  image_paths = sorted(glob(args.image_pattern))
  logging.debug('image_paths:\n%s' % pformat(image_paths, indent=2))
  if not image_paths:
    raise IOError('Image files do not exist for the frame pattern: %s' % args.image_pattern)
  mask_paths = sorted(glob(args.mask_pattern))
  logging.debug('mask_paths:\n%s' % pformat(mask_paths, indent=2))

  # Group them into pairs.
  images = [(op.splitext(op.basename(image_path))[0], image_path) for image_path in image_paths]
  masks  = [(op.splitext(op.basename(mask_path))[0], mask_path) for mask_path in mask_paths]

  # Group those that match and unmatched image_paths first.
  pairs  = []
  for image_name, image_path in images:
    mask_ids = [mask_id for mask_id, (mask_name, _) in enumerate(masks) if mask_name == image_name]
    assert len(mask_ids) <= 1, (image_name, mask_ids)
    if len(mask_ids) == 1:
      mask_id = mask_ids[0]
      mask_path = masks[mask_id][1]
      del masks[mask_id]
      pairs.append((image_path, mask_path))

    else:
      pairs.append((image_path, None))
  logging.debug('Pairs:\n%s' % pformat(pairs, indent=2))

  if len(masks) > 0:
    logging.warning('%d masks dont have their images, e.g. %s' % (len(masks), masks[0][1]))

  logging.info('Found %d image pictures, %d of them have masks' %
      (len(pairs), len([x for x in pairs if x[1] is not None])))

  # Write to database.
  relpath = os.getenv('HOME')
  for image_path, mask_path in pairs:
    image = cv2.imread(image_path)
    h, w = image.shape[0:2]
    if mask_path is not None:
      mask = cv2.imread(mask_path)
      if not mask.shape[0:2] == (h, w):
        logging.warning('Image %s and mask %s have different dimensions: %s vs %s' %
            (image_path, mask_path, (h, w), mask.shape[0:2]))
      maskfile = op.relpath(op.abspath(mask_path), relpath)
    else:
      maskfile = None
    imagefile = op.relpath(op.abspath(image_path), relpath)
    s = 'INSERT INTO images(imagefile,maskfile,src,width,height) VALUES (?,?,?,?,?)'
    c.execute(s, (imagefile, maskfile, args.src, w, h))

