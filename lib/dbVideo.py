import os, sys, os.path as op
import numpy as np
import cv2
import logging
from progressbar import progressbar
from backendImages import imread, VideoWriter
from utilities import drawScoredRoi


def add_parsers(subparsers):
  writeVideoParser(subparsers)


def writeVideoParser(subparsers):
  parser = subparsers.add_parser('writeVideo', description='Write video of "imagefile" entries.')
  parser.add_argument('--out_videofile', required=True)
  parser.add_argument('--fps', type=int, default=2)
  parser.add_argument('--overwrite', action='store_true', help='overwrite video if it exists.')
  parser.add_argument('--with_frameid', action='store_true', help='print frame number.')
  parser.add_argument('--with_boxes', action='store_true', help='draw bounding boxes.')
  parser.add_argument('--with_car_info', action='store_true', help='print name, yaw, pitch, score on the image.')
  parser.set_defaults(func=writeVideo)

def writeVideo (c, args):
  logging.info ('==== writeVideo ====')
  font = cv2.FONT_HERSHEY_SIMPLEX

  writer = VideoWriter(vimagefile=args.out_videofile, overwrite=args.overwrite, fps=args.fps)

  c.execute('SELECT imagefile,height FROM images')
  for imagefile,imheight in progressbar(c.fetchall()):

    frame = imread(imagefile).copy()

    if args.with_frameid:
      fontscale = float(imheight) / 700
      thickness = imheight / 700
      offsety = imheight / 30
      cv2.putText (frame, op.basename(imagefile), (10, 10 + offsety), font, fontscale, (0,0,0), thickness=thickness*3)
      cv2.putText (frame, op.basename(imagefile), (10, 10 + offsety), font, fontscale, (255,255,255), thickness=thickness)

    c.execute('SELECT * FROM cars WHERE imagefile=?', (imagefile,))
    for car_entry in c.fetchall():
      roi   = carField (car_entry, 'roi')
      name  = carField (car_entry, 'name')
      score = carField (car_entry, 'score')
      logging.debug ('roi: %s, score: %s' % (str(roi), str(score)))

      if args.with_boxes:
        drawScoredRoi (frame, roi, label=name, score=score)

      if args.with_car_info:
        name  = carField(car_entry, 'name')
        color = carField(car_entry, 'color')
        score = carField(car_entry, 'score')
        yaw   = carField(car_entry, 'yaw')
        pitch = carField(car_entry, 'pitch')
        #cv2.putText (frame, '%.0f' % yaw, (05, 50), font, 0.5, (255,255,255), 2)
        cv2.putText (frame, 'name: %s' % name, (10 + roi[1], 20 + roi[0]), font, 0.5, (255,255,255), 2)
        if color is not None:
          cv2.putText (frame, 'color: %s' % color, (10 + roi[1], 40 + roi[0]), font, 0.5, (255,255,255), 2)
        if score is not None:
          cv2.putText (frame, 'score: %.3f' % score, (10 + roi[1], 60 + roi[0]), font, 0.5, (255,255,255), 2)
        if yaw is not None:
          cv2.putText (frame, 'yaw: %.1f' % yaw, (10 + roi[1], 80 + roi[0]), font, 0.5, (255,255,255), 2)
        if pitch is not None:
          cv2.putText (frame, 'pitch: %.1f' % pitch, (10 + roi[1], 100 + roi[0]), font, 0.5, (255,255,255), 2)

    writer.imwrite(frame)

  writer.close()
