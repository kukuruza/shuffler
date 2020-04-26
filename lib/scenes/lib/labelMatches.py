#!/usr/bin/env python
import os, os.path as op
import logging
import shutil
import argparse
import numpy as np
from time import time
import cv2
import simplejson as json
import colorsys
from pprint import pprint, pformat
from scipy.misc import imread
from lib.cvWindow import Window
from lib.warp import warp, transformPoint


def get_random_color():
  ''' Get a random bright color. '''
  h,s,l = np.random.random(), 0.5 + np.random.random()/2.0, 0.4 + np.random.random()/5.0
  r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
  return r,g,b


def getGifFrame(img1, img2, frac):
  k = np.sin(frac * np.pi * 2.) / 2.
  return (img1.astype(float) * (0.5 + k) +
          img2.astype(float) * (0.5 - k)).astype(np.uint8)


class MatchWindow(Window):
  ''' Use mouse left button and wheel to navigate, shift + left button
  to select points. Select a point in each image, and the match will be added.
  Alt + left button on a point in either image to delete a match.
  '''

  def __init__(self, img, winsize=500, name='display'):
    self.pointselected = None
    self.pointdeleted = None
    self.points = []  # (x, y), (b,g,r) in input image coordinate system.
    Window.__init__(self, img, winsize, name)

  def mouseHandler(self, event, x, y, flags, params):

    # Call navigation handler from the base class.
    Window.mouseHandler(self, event, x, y, flags, params)

    # Select a point.
    if event == cv2.EVENT_LBUTTONDOWN and flags == 17:  # Shift
      logging.info('%s: registered mouse select press.' % self.name)
      x, y = self.window_to_image_coords(x, y)
      self.pointselected = (x, y)
      self.update_cached_zoomed_img()
      self.redraw()

    # Delete a point.
    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_ALTKEY):
      logging.info('%s: registered mouse delete press at %d %d.' % (self.name, x, y))
      x, y = self.window_to_image_coords(x, y)
      self.pointdeleted = self._any_point_selected(x, y)  # None if not found.

  def update_cached_zoomed_img(self):
    Window.update_cached_zoomed_img(self)
    for (x, y), color in self.points:
      self._drawpoint(x, y, color)
    if self.pointselected is not None:
      self._drawpoint(self.pointselected[0], self.pointselected[1], (0,0,255))

  def _drawpoint(self, x, y, color):
    x, y = self.image_to_zoomedimage_coords(x, y)
    cv2.circle (self.cached_zoomed_img, (int(x), int(y)), 10, color, thickness=3)

  def _any_point_selected(self, xsel, ysel):
    SELECT_DIST = 5
    iselected = None
    for ip, ((x, y), _) in enumerate(self.points):
      if  abs(xsel - x) < SELECT_DIST and abs(ysel - y) < SELECT_DIST:
        logging.info('Deleted point %d.' % ip)
        return ip
    return None


class GifWindow:
  def __init__(self):
    self.img1 = self.img2 = None
    self.start = time()
    self.CYCLE_S = 2.0

  def update_images(self, img1, img2):
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    self.img1 = img1.astype(float)
    self.img2 = img2.astype(float)

  def redraw(self):
    if self.img1 is not None:
      k = (time() - self.start) / self.CYCLE_S
      img = getGifFrame(self.img1, self.img2, k)
      cv2.imshow('warped', img)


def _redraw(window1, window2, gif_window):
  window1.update_cached_zoomed_img()
  window2.update_cached_zoomed_img()
  window1.redraw()
  window2.redraw()
  # If at least 4 points are present, we can compute H from window1 to window2.
  if len(window1.points) >= 4:
    # Points have (x, y) order.
    src_pts = np.asarray([p[0] for p in window1.points])
    dst_pts = np.asarray([p[0] for p in window2.points])
    H, matches_mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
    #matches_mask = matches_mask.ravel().tolist()
    warped = warp(window1.img, H, None, window2.img.shape[0:2])
    dst = window2.img.copy()
    warped_pts = np.asarray([transformPoint(H, pt[0], pt[1]) for pt in src_pts])
    for (x1, y1), (x2, y2) in zip(warped_pts.astype(int), dst_pts.astype(int)):
      cv2.arrowedLine(warped, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
      cv2.arrowedLine(dst, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
    gif_window.update_images(dst, warped)
    #cv2.imshow('warped', warped)


def labelMatches (img1, img2, matches_path, 
                  winsize1=500, winsize2=500, 
                  backup_matches=True, name1='frame', name2='map'):

  window1 = MatchWindow(img1, winsize1, name=name1)
  window2 = MatchWindow(img2, winsize2, name=name2)
  gif_window = GifWindow()

  # If already exists, we'll load existing matches.
  # pts_pairs is a list of tuples (x1, y1, x2, y2)
  if op.exists(matches_path):
    if backup_matches:
      backup_path = op.splitext(matches_path)[0] + '.backup.json'
      shutil.copyfile(matches_path, backup_path)
    with open(matches_path) as f:
      matches = json.load(f)
    for i in range(len(matches[name1]['x'])):
      color = get_random_color()
      window1.points.append(((matches[name1]['x'][i], matches[name1]['y'][i]), color))
      window2.points.append(((matches[name2]['x'][i], matches[name2]['y'][i]), color))
  _redraw(window1, window2, gif_window)

  BUTTON_ESCAPE = 27
  BUTTON_ENTER = 13
  button = -1
  redraw = False
  while button != BUTTON_ESCAPE and button != BUTTON_ENTER:

    if window1.pointselected is not None and window2.pointselected is not None:
      logging.info('Adding a match')
      color = get_random_color()
      window1.points.append((window1.pointselected, color))
      window2.points.append((window2.pointselected, color))
      window1.pointselected = None
      window2.pointselected = None
      _redraw(window1, window2, gif_window)
    elif window1.pointdeleted is not None or window2.pointdeleted is not None:
      logging.info('Deleting a match')
      if window1.pointdeleted is not None:
        ideleted = window1.pointdeleted
      elif window2.pointdeleted is not None:
        ideleted = window2.pointdeleted
      else:
        assert 0
      del window1.points[ideleted]
      del window2.points[ideleted]
      window1.pointdeleted = None
      window2.pointdeleted = None
      _redraw(window1, window2, gif_window)

    gif_window.redraw()
    button = cv2.waitKey(50)

  # Save and exit.
  if button == BUTTON_ENTER:
    matches = {name1: {'x': [], 'y': []}, name2: {'x': [], 'y': []}}
    for p in window1.points:
      matches[name1]['x'].append(p[0][0])
      matches[name1]['y'].append(p[0][1])
    for p in window2.points:
      matches[name2]['x'].append(p[0][0])
      matches[name2]['y'].append(p[0][1])
    if not op.exists(op.dirname(matches_path)):
      os.makedirs(op.dirname(matches_path))
    with open(matches_path, 'w') as f:
      f.write(json.dumps(matches, sort_keys=True, indent=2))
  elif button == BUTTON_ESCAPE:
    logging.info('Exiting without saving.')


def loadMatches(matches_path, name1, name2):

  # Load matches.
  assert op.exists(matches_path), matches_path
  matches = json.load(open(matches_path))
  logging.debug (pformat(matches, indent=2))
  src_pts = matches[name1]
  dst_pts = matches[name2]
  assert range(len(src_pts['x'])) == range(len(dst_pts['x']))
  assert range(len(src_pts['x'])) == range(len(src_pts['y']))
  assert range(len(dst_pts['x'])) == range(len(dst_pts['y']))

  # Matches as numpy array.
  N = range(len(src_pts['x']))
  src_pts = np.float32([ [src_pts['x'][i], src_pts['y'][i]] for i in N ])
  dst_pts = np.float32([ [dst_pts['x'][i], dst_pts['y'][i]] for i in N ])
  logging.debug(src_pts)
  logging.debug(dst_pts)
  return src_pts, dst_pts


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-1', '--image_path1', required=True)
  parser.add_argument('-2', '--image_path2', required=True)
  parser.add_argument('--matches_path', required=True)
  parser.add_argument('--winsize1', type=int, default=500)
  parser.add_argument('--winsize2', type=int, default=500)
  parser.add_argument('--name1', default='frame')
  parser.add_argument('--name2', default='map')
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  assert op.exists(image_path1), image_path1
  assert op.exists(image_path2), image_path2
  img1 = imread(image_path1)
  img2 = imread(image_path2)
  assert img1 is not None
  assert img2 is not None

  labelMatches (img1, img2, args.matches_path,
      winsize1=args.winsize1, winsize2=args.winsize2,
      name1=args.name1, name2=args.name2)
