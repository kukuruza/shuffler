import sys, os, os.path as op
import logging
import argparse
import numpy as np
import cv2
from scipy.misc import imread, imresize


def pad_to_square(img):
  ''' Pads the right or the bottom of an image to make it square. '''
  assert img is not None
  pad = abs(img.shape[0] - img.shape[1])
  if img.shape[0] < img.shape[1]:
    img = np.pad(img, ((0,pad),(0,0),(0,0)), 'constant')
  else:
    img = np.pad(img, ((0,0),(0,pad),(0,0)), 'constant')
  return img

def clip(x, xmin, xmax):
  return max(min(x, xmax), xmin)


class Window:

  def __init__(self, img, winsize=500, name='display', num_zoom_levels=10):
    self.name = name
    self.img = pad_to_square(img.copy()[:,:,::-1])  # cv2 expects image as BGR.
    #
    self.imgsize = self.img.shape[0]
    self.winsize = winsize
    logging.debug('%s: imgsize: %d, winsize: %d' %
        (self.name, self.imgsize, self.winsize))
    self.Max_Zoom = self.imgsize / float(self.winsize)
    logging.info('%s: max_zoom: %f' % (self.name, self.Max_Zoom))
    assert num_zoom_levels > 1, num_zoom_levels  
    self.Num_Zoom_Levels = num_zoom_levels if self.Max_Zoom > 1 else 1
    self.Zoom_levels = range(self.Num_Zoom_Levels)
    logging.info('%s: zooms_levels: %d' % (self.name, self.Num_Zoom_Levels))
    self.zoom_level = 0.
    self.scrollx = 0.5  # 0. to 1.
    self.scrolly = 0.5  # 0. to 1.
    #
    self.rbuttonpressed = False
    self.rpressx = None
    self.rpressy = None
    #
    self.zbuttonpressed = False
    self.zpressy = None
    #
    self.cached_zoomed_img = None
    #
    cv2.namedWindow(self.name)
    cv2.setMouseCallback(self.name, self.mouseHandler)
    #
    self.make_cached_zoomed_images()

  def mouseHandler(self, event, x, y, flags, params):

    # Zooming.
    if event == cv2.EVENT_MOUSEWHEEL:
      logging.debug('%s: registered mouse zooming press.' % self.name)
      self.zoom_level += (1 if flags > 0 else -1)
      self.zoom_level = clip(self.zoom_level, 0, len(self.Zoom_levels) - 1)
      self.update_cached_zoomed_img()
      self.redraw()

    # Scrolling.
    else:
      if event == cv2.EVENT_LBUTTONDOWN:
        logging.debug('%s: registered mouse scroll press.' % self.name)
        self.rbuttonpressed = True
        self.rpressx, self.rpressy = x, y
      elif event == cv2.EVENT_LBUTTONUP:
        self.rbuttonpressed = False
        self.rpressx, self.rpressy = None, None
        logging.debug('%s: released mouse scroll press.' % self.name)
      elif event == cv2.EVENT_MOUSEMOVE:
        if self.rbuttonpressed:
          if self.cropsize != self.winsize:
            self.scrollx -= (x - self.rpressx) / float(self.cropsize - self.winsize)
            self.scrolly -= (y - self.rpressy) / float(self.cropsize - self.winsize)
            self.scrollx = clip(self.scrollx, 0., 1.)
            self.scrolly = clip(self.scrolly, 0., 1.)
          self.rpressx, self.rpressy = x, y
          self.redraw()

  def get_zoom(self, zoom_level):
    if self.Num_Zoom_Levels == 1:
      return 1. / self.Max_Zoom
    else:
      return ((float(self.Max_Zoom) - 1.) / (self.Num_Zoom_Levels - 1.) * zoom_level + 1.) / self.Max_Zoom

  def make_cached_zoomed_images(self):
    ''' Cache images at different zoom levels to make scrolling faster. '''
    self.cached_zoomed_images = {}
    for zoom_level in self.Zoom_levels:
      zoom = self.get_zoom(zoom_level)
      zoomed_img = cv2.resize(self.img, (0,0), fx=zoom, fy=zoom)  # Scipy resizes very slowly.
      logging.debug('%s: caching zoom level %.2f, zoom %.2f, imsize %s' %
          (self.name, zoom_level, zoom, zoomed_img.shape))
      self.cached_zoomed_images[zoom_level] = zoomed_img

  def update_cached_zoomed_img(self):
    ''' Pick one from the one the image pyramid. '''
    self.cached_zoomed_img = self.cached_zoomed_images[int(self.zoom_level)].copy()
    self.cropsize = self.cached_zoomed_img.shape[0]
    logging.debug('%s: got cached image of shape %s' % (self.name, self.cropsize))

  def get_offsets(self):
    ''' Get win offsets based on zoom and scrolls '''
    assert self.cached_zoomed_img is not None
    maxoffset = self.cropsize - self.winsize
    logging.debug ('%s: zoom level: %.1f, cropsize: %d, maxoffset: %d' %
        (self.name, self.zoom_level, self.cropsize, maxoffset))
    offsetx = int(self.scrollx * maxoffset)
    offsety = int(self.scrolly * maxoffset)
    assert offsetx >= 0 and offsetx <= maxoffset, offsetx
    assert offsety >= 0 and offsety <= maxoffset, offsety
    return offsetx, offsety
    
  def redraw(self):
    offsetx, offsety = self.get_offsets()
    crop = self.cached_zoomed_img
    logging.debug('%s: redraw offset: %d, %d, cropsize: %d %d' %
        (self.name, offsetx, offsety, crop.shape[1], crop.shape[0]))
    crop = crop[offsety : offsety + self.winsize,
                offsetx : offsetx + self.winsize, :]
    cv2.imshow(self.name, crop)

  def image_to_zoomedimage_coords(self, x, y):
    zoom = self.get_zoom(self.zoom_level)
    xwin = x * zoom
    ywin = y * zoom
    logging.debug('%s: img2win zoom: %.1f;  %.1f, %.1f -> %.1f, %.1f' %
        (self.name, zoom, x, y, xwin, ywin))
    return xwin, ywin

  def window_to_image_coords(self, x, y):
    offsetx, offsety = self.get_offsets()
    zoom = self.get_zoom(self.zoom_level)
    xim = float(x + offsetx) / zoom
    yim = float(y + offsety) / zoom
    logging.debug('%s: win2img offset: %d, %d, zoom: %.1f;  %.1f, %.1f -> %.1f, %.1f' %
        (self.name, offsetx, offsety, zoom, x, y, xim, yim))
    return xim, yim



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--image_path', required=True)
  parser.add_argument('--winsize', type=int, default=500)
  parser.add_argument('--num_zoom_levels', type=int, default=20)
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  window = Window(imread(args.image_path), winsize=args.winsize, 
      num_zoom_levels=args.num_zoom_levels)
  window.update_cached_zoomed_img()
  window.redraw()
  while cv2.waitKey(50) != 27:
    pass
  

