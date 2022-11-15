import numpy as np
import cv2
import logging

class AzimuthWindow(Window):
  ''' Use Shift + left button to choose azimuth. '''

  def __init__(self, img, x, y, axis_x, axis_y, winsize=500):
    Window.__init__(self, img, winsize, name='azimuth', num_zoom_levels=2)
    self.is_just_a_suggestion = False  # Used to pick color.
    self.yaw = None
    self.selected = False
    self.x, self.y = self.image_to_zoomedimage_coords(x, y)
    self.axis_x = axis_x * self.get_zoom(self.zoom_level)
    self.axis_y = axis_y * self.get_zoom(self.zoom_level)

  def mouseHandler(self, event, x, y, flags, params):

    # Call navigation handler from the base class.
    Window.mouseHandler(self, event, x, y, flags, params)

    # Display and maybe select azimuth.
    #   flags == 16 <=> Shift + no mouse press
    #   flags == 17 <=> Shift + left button down
    if (event == cv2.EVENT_MOUSEMOVE and flags == 16 or
        event == cv2.EVENT_LBUTTONDOWN and flags == 17):
      logging.debug('%s: registered shift + mouse move and maybe press.' % self.name)
      self.is_just_a_suggestion = False
      # 0 is north, 90 is East.
      self.yaw = (np.arctan2((x - self.x) / self.axis_x, 
                            -(y - self.y) / self.axis_y) * 180. / np.pi) % 360
      logging.debug('%s, yaw is at %0.f' % (self.name, self.yaw))
      self.update_cached_zoomed_img()
      self.redraw()
      if event == cv2.EVENT_LBUTTONDOWN:
        self.selected = True
        logging.debug('%s: registered shift + mouse press.' % self.name)

  def update_cached_zoomed_img(self):
    Window.update_cached_zoomed_img(self)
    color = (255,0,0)
    cv2.ellipse(self.cached_zoomed_img, 
        (int(self.x), int(self.y)), (int(self.axis_x * 0.6), int(self.axis_y * 0.6)),
        startAngle=0, endAngle=360, angle=0, color=color, thickness=2)
    if self.yaw:
      y1 = self.y - self.axis_y * np.cos(self.yaw * np.pi / 180.) * 1.2
      x1 = self.x + self.axis_x * np.sin(self.yaw * np.pi / 180.) * 1.2
      cv2.arrowedLine(self.cached_zoomed_img,
          (int(self.x),int(self.y)), (int(x1),int(y1)), color=color, thickness=2)
      postfix = '(suggested by azimuth map)' if self.is_just_a_suggestion else ''
      cv2.putText(self.cached_zoomed_img, 'yaw %.0f %s' % (self.yaw, postfix), (10, 70),
          cv2.FONT_HERSHEY_SIMPLEX, 1., color, 2)


def _getFlatteningFromImagefile(poses, imagefile, y_frame, x_frame):
  H = getHfromPose(poses[imagefile])
  if H is not None:
    return getFrameFlattening(H, y_frame, x_frame)
  else:
    return 1
