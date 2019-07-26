import math
import numpy as np

def crop_frame(frame, x, y, width, height, is_color=False):
  """
  Crops a OpenCV2 frame

  Will add borders to edges in order to maintain width and height
  even when cropping near edges.

  Arguments:
    frame ndarray The numpy multi-d array representing the image
    x int The center of the cropped image, in pixels
    y int The center of the cropped image, in pixels
    width int The width of the cropped image, in pixels
    height int The height of the cropped image, in pixels
    is_color bool Whether the image is a color image

  Returns:
    ndarray The cropped image
  """
  y_radius = int(math.floor(height/2))
  x_radius = int(math.floor(width/2))

  if frame.shape[0] < height:
    offset = height-frame.shape[0]
    border_size = ( offset, frame.shape[1], 3 ) if is_color else ( offset, frame.shape[1] )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((border, frame), axis=0)
    y += offset

  if frame.shape[1] < width:
    offset = width-frame.shape[1]
    border_size = ( frame.shape[0], offset, 3 ) if is_color else ( frame.shape[0], offset )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((border, frame), axis=1)
    x += offset

  # Check our bounds
  if y-y_radius < 0:
    # We need to add a border to the top
    offset = abs(y-y_radius)
    border_size = ( offset, len(frame[0]), 3 ) if is_color else ( offset, len(frame[0]) )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((border, frame), axis=0)

    y += offset # What was (0, 0) is now (0, [offset])

  if y+y_radius > frame.shape[0]-1:
    # We need to add a border to the bottom
    offset = abs(frame.shape[0]-1-y_radius)
    border_size = ( offset, len(frame[0]), 3 ) if is_color else ( offset, len(frame[0]) )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((frame, border), axis=0)
    # What was (0, 0) is still (0, 0)

  if x-x_radius < 0:
    # We need to add a border to the left
    offset = abs(x-x_radius)
    border_size = ( len(frame), offset, 3 ) if is_color else ( len(frame), offset )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((border, frame), axis=1)

    x += offset # What was (0, 0) is now ([offset], 0)

  if x+x_radius > frame.shape[1]-1:
    # We need to add a border to the left
    offset = abs(frame.shape[1]-1-x_radius)
    border_size = ( len(frame), offset, 3 ) if is_color else ( len(frame), offset )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((frame, border), axis=1)
    # What was (0, 0) is still (0, 0)

  left = x-x_radius + (width-2*x_radius) # To account for rounding errors
  right = x+x_radius
  top = y-y_radius + (width-2*y_radius)
  bottom = y+y_radius
  frame = frame[top:bottom, left:right]

  return frame