import math
import numpy as np
import cv2
from tqdm import tqdm

FONT           = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONT_SCALE     = 1
FONT_COLOR     = (255,255,255)
FONT_LINE_TYPE = 2

# BGR
CIRCLE_COLORS     = {
  "N": (50,50,50),
  "R": (100,100,255),
  "E": (100,255,100),
  "X": (255,255,255),
  "M": (255,100,100),
  "?": (100,100,255)
}
CIRCLE_THICKNESS  = {
  "N": 1,
  "R": 2,
  "E": 2,
  "X": 2,
  "M": 2,
  "?": 2
}
CIRCLE_RADIUS     = 30
CIRCLE_LINE_TYPE  = cv2.LINE_AA


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

def make_videos(data, output_path, annotate=True, draw_tracks=False, codec='mp4v'):
  for data_set in data['data_set'].unique():
    ds_data = data[( (data['data_set'] == data_set) )].copy()
    if ds_data.shape[0] <= 0:
      continue

    ds_data = ds_data[[ 'data_set', 'particle_id', 'frame', 'time', 'x_px', 'y_px', 'event' ]]
    ds_data.sort_values('frame')

    ds_data.loc[:,'frame_path'] = ds_data.apply(lambda x: str( (tiff_path / (data_set + '/' + str(x.frame).zfill(4) + '.tif')).resolve() ), axis=1)
    
    field_video_path = output_path / (data_set + '.mp4')
    make_video(ds_data, str(field_video_path), movie_name = data_set, annotate=annotate, draw_tracks=draw_tracks, codec=codec)

    for particle_id in ds_data['particle_id'].unique():
      p_data = ds_data[(ds_data['particle_id'] == particle_id)].copy()
      if p_data.shape[0] <= 0:
        continue

      (output_path / data_set).mkdir(mode=0o755, parents=True, exist_ok=True)

      r_graph_gen_path = (ROOT_PATH / ("lib/R/make-single-cell-graph.R")).resolve()
      graph_path = output_path / (data_set + "/" + particle_id + ".tif")
      # Make our graph
      cmd = [
        "Rscript",
        "--vanilla",
        str(r_graph_gen_path),
        str(output_file_path),
        str(graph_path),
        data_set,
        particle_id,
        "300"
      ]
      subprocess.call(cmd)

      p_data.loc[:,'graph_path'] = str(graph_path)

      cell_video_path = output_path / (data_set + "/" + particle_id + ".mp4")

      make_video(p_data, str(cell_video_path), crop=( 100, 100 ), scale=3.0, movie_name = data_set + "/" + particle_id, annotate=annotate, draw_tracks=draw_tracks, codec=codec)

      if graph_path.exists():
        graph_path.unlink()

def append_graph(frame, graph, this_frame_i, start_frame_i, end_frame_i):
  # Draw a line on the graph
  gc = np.copy(graph)
  x = int(round(gc.shape[1]*(this_frame_i-start_frame_i)/(end_frame_i-start_frame_i)))
  gc = cv2.line(gc, (x, 0), (x, gc.shape[0]), (255, 255, 255))
  frame = np.concatenate((frame, gc), axis=0)

  return frame

def make_video(data, output_file_path, annotate=True, draw_tracks=False, crop=False, scale=1.0, movie_name=None, codec='mp4v'):
  particle_ids = data['particle_id'].unique()

  if len(particle_ids) > 1 and crop is not False:
    raise ValueError('Cannot crop video if there is more than one cell present in data')

  data_set = data['data_set'].iloc[0]

  if not crop:
    first_frame = cv2.imread(str(data['frame_path'].iloc[0]), cv2.IMREAD_GRAYSCALE)
    width = first_frame.shape[1]
    height = first_frame.shape[0]
  else:
    width = crop[0]
    height = crop[1]

  movie_width = int(width*scale)
  movie_height = int(height*scale)

  if 'graph_path' in data:
    first_graph = cv2.imread(str(data['graph_path'].iloc[0]), cv2.IMREAD_COLOR)
    movie_height += first_graph.shape[0]

  # Create a blank frame in case we have missing frames
  zero_frame = np.zeros(( height, width ))
  zero_frame = zero_frame.astype('uint8')

  fourcc = cv2.VideoWriter_fourcc(*codec)
  writer = cv2.VideoWriter(str(output_file_path), fourcc, 10, (movie_width, movie_height), True)

  if draw_tracks:
    track_frame = cv2.cvtColor(zero_frame, cv2.COLOR_GRAY2BGR)

    for particle_id in particle_ids:
      p_data = data[((data['particle_id'] == particle_id))]
      prev_x = None
      prev_y = None
      for index, row in p_data.iterrows():
        x = row['x_px']
        y = row['y_px']
        if prev_x is not None:
          cv2.line(track_frame, (prev_x, prev_y), (x, y), (255, 255, 255), 1, CIRCLE_LINE_TYPE)
        prev_x = x
        prev_y = y

  start_frame_i = np.min(data['frame'])
  end_frame_i = np.max(data['frame'])
  description = 'Building movie'
  if movie_name is not None:
    description += " for " + movie_name + ""
  for frame_i in tqdm(range(start_frame_i, ( end_frame_i+1 )), desc=description, unit='frames'):
    f_data = data[(data['frame'] == frame_i)]

    frame = zero_frame
    if f_data['frame'].count() > 0: # We're not missing a frame
      frame_path = f_data['frame_path'].iloc[0]

      frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

      if crop is not False:
        x = f_data['x_px'].iloc[0]
        y = f_data['y_px'].iloc[0]

        frame = crop_frame(frame, x, y, width, height)

      if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        frame = frame.astype('uint8')

      # Make the frame color
      frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

      if annotate:
        if len(particle_ids) > 1:
          for index, row in f_data.iterrows():
            particle_id = row['particle_id']
            x = row['x_px']
            y = row['y_px']
            event = row['event'] if 'event' in row else 'N'

            # Add particle_id
            adj_circle_radius = int(round(CIRCLE_RADIUS))
            cv2.circle(frame, (x, y), adj_circle_radius, CIRCLE_COLORS[event], CIRCLE_THICKNESS[event], CIRCLE_LINE_TYPE)
            cv2.putText(frame, particle_id, (x, y), FONT, FONT_SCALE, FONT_COLOR, FONT_LINE_TYPE)

        # Make frame text
        hours = math.floor(f_data['time'].iloc[0] / 3600)
        minutes = math.floor((f_data['time'].iloc[0] - (hours*3600)) / 60)
        seconds = math.floor((f_data['time'].iloc[0] - (hours*3600)) % 60)

        time_label = "{:02d}h{:02d}'{:02d}\" ({:d})".format(hours, minutes, seconds, frame_i)

        cv2.putText(frame, time_label, (10, 20), FONT, FONT_SCALE, FONT_COLOR, FONT_LINE_TYPE)

      if draw_tracks:
        frame = cv2.addWeighted(track_frame, 0.2, frame, 0.8, 0)

      if 'graph_path' in data:
        graph = cv2.imread(str(f_data['graph_path'].iloc[0]))
        frame = append_graph(frame, graph, frame_i, start_frame_i, end_frame_i)

      writer.write(frame)
  writer.release()




