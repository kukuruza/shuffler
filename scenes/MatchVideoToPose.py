#!/usr/bin/env python
import os, os.path as op
import logging
import argparse
import cv2
from imageio import imwrite, get_writer
from pkg_resources import parse_version
from lib.labelMatches import labelMatches
from lib.scene import Video, _atcity


# returns OpenCV VideoCapture property id given, e.g., "FPS"
def _capPropId(prop):
  OPCV3 = parse_version(cv2.__version__) >= parse_version('3')
  return getattr(cv2 if OPCV3 else cv2.cv, ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description=
      '''Match a frame from a video to the closest pose''')
  parser.add_argument('--video_file', required=True, type=str,
      help='The path to the video .avi file. Will load a frame from it.')
  parser.add_argument('--video_frame_id', type=int, default=0,
      help='Sometimes the first video frame is not best for fitting.')
  parser.add_argument('--no_backup', action='store_true')
  parser.add_argument('--winsize1', type=int, default=700)
  parser.add_argument('--winsize2', type=int, default=700)
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  # Load pose and its example frame.
  video = Video.from_imagefile(args.video_file)
  assert video is not None, 'Failed to load video json info.'
  poseframe = video.pose.load_example()

  # Load a frame from a video.
  video_path = _atcity(args.video_file)
  assert op.exists(video_path), video_path
  handle = cv2.VideoCapture(video_path)
  assert handle, 'Video failed to open: %s' % handle
  video_length = int(handle.get(_capPropId('FRAME_COUNT')))
  assert args.video_frame_id < video_length
  handle.set(_capPropId('POS_FRAMES'), args.video_frame_id)
  retval, videoframe = handle.read()
  videoframe = videoframe[:,:,::-1]
  if not retval:
    raise Exception('Failed to read the frame.')

  # Get the matches path.
  video_name = op.splitext(op.basename(args.video_file))[0]
  matches_path = op.join(video.get_video_dir(), 'matches-pose%d.json' % video.pose.pose_id)

  labelMatches (videoframe, poseframe, matches_path,
      winsize1=args.winsize1, winsize2=args.winsize2,
      name1='video', name2='pose',
      backup_matches=not args.no_backup)

  if op.exists(video.get_video_dir()):
    videoframe_path = op.join(video.get_video_dir(), 'example.jpg')
    imwrite(videoframe_path, videoframe)