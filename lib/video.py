import math
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from errors import NoImagesFound

from skimage.color import label2rgb

# BGR
CIRCLE_COLORS     = {
  "N": 'rgb(50,50,50)',
  "R": 'rgb(100,100,255)',
  "E": 'rgb(100,255,100)',
  "X": 'rgb(255,255,255)',
  "M": 'rgb(255,100,100)',
  "?": 'rgb(100,100,255)'
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

def make_videos(tiff_path, data, output_path, annotate=True, draw_tracks=True, codec='mp4v'):
  for data_set in data['data_set'].unique():
    ds_data = data[( (data['data_set'] == data_set) )].copy()
    if ds_data.shape[0] <= 0:
      continue

    ds_data = ds_data[[ 'data_set', 'particle_id', 'frame', 'time', 'x_px', 'y_px', 'event' ]]
    ds_data.sort_values('frame')
    
    field_video_path = output_path / (data_set + '.mp4')
    data_set_tiff_path = tiff_path / data_set
    data_set_mask_path = data_set_tiff_path / "tracks"
    frame_rate = ds_data['frame_rate'].iloc[0]
    make_video(
      data_set_tiff_path, 
      data_set_mask_path, 
      field_video_path, 
      data=ds_data, 
      frame_rate=frame_rate, 
      movie_name = data_set, 
      annotate=annotate, 
      draw_tracks=draw_tracks, 
      codec=codec
    )

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
      make_video(
        data_set_tiff_path, 
        data_set_mask_path, 
        cell_video_path, 
        data=p_data, 
        frame_rate=frame_rate, 
        movie_name = data_set + "/" + particle_id, 
        annotate=False, 
        draw_tracks=False, 
        codec=codec
      )

      if graph_path.exists():
        graph_path.unlink()

def make_video(tiff_path, mask_path, output_file_path, graph_path=None, frame_rate=180, draw_tracks=True, data=None, annotate=True, crop=False, scale=1.0, movie_name=None, codec='mp4v', lut='glasbey'):
  lut = pd.read_csv(str(ROOT_PATH / ("lib/luts/" + lut + ".lut")), dtype={'red': int, 'green': int, 'blue': int})
  # label2rgb expects colors in a [0,1] range
  lut['red_float'] = lut['red']/255
  lut['green_float'] = lut['green']/255
  lut['blue_float'] = lut['blue']/255

  image_files = list(tiff_path.glob("*.tif"))
  image_files.sort(key=lambda x: str(len(str(x))) + str(x))
  if len(image_files) < 1:
    raise NoImagesFound('No images found in ' + str(tiff_path))

  description = 'Building movie'
  if movie_name is not None:
    description += " for " + movie_name + ""

  top_padding = 25
  bottom_padding = 0
  track_frame = None
  old_props = None
  annotated_frames = []
  
  for image_file in tqdm(image_files, desc=description):
    image = Image.open(str(image_file))
    if not (mask_path / image_file.name).exists():
      raise ValueError("No mask matches " + str(image_file) + " in " + str(mask_path))
    mask = Image.open(str(tracks_file))

    # Assign colors to each label
    props = pd.DataFrame(regionprops_table(mask, properties=('label', 'centroid')))
    props.rename(columns={ 'centroid-0': 'y', 'centroid-1': 'x' }, inplace=True)
    if props.shape[0] > lut.shape[0]:
      new_luts = lut.sample(props.shape[0]-lut.shape[0], replace=True)
      lut = pd.concat(lut, new_luts)
    props = props.merge(lut, left_on='label', right_index=True)

    # Add mask overlay
    r = props['red_float'].tolist()
    g = props['green_float'].tolist()
    b = props['blue_float'].tolist()
    colors = list(zip(r, g, b))

    np_image = label2rgb(np.asarray(mask), image=np.asarray(image), colors=colors, alpha=0.4)
    image = Image.fromarray((np_image*255).astype(np.uint8))

    # Annotate the frame if needed
    frame_idx = int(image_file.stem)
    if annotate and data is not None:
      image = _annotate_image(image, data.loc[(data['frame'] == frame_idx)])

    # Crop the frame if need be
    if crop is not False and data is not None:
      draw_tracks = False
      crop_x = data[(data['frame'] == frame_idx), 'x_px'].iloc[0]
      crop_y = data[(data['frame'] == frame_idx), 'y_px'].iloc[0]

      image = Image.fromarray(crop_frame(np.asarray(image), crop_x, crop_y, crop[0], crop[1]))

    if scale != 1.0:
      image = image.resize((scale*image.size[0], scale*image.size[1]), resample=Image.BILINEAR)

    # Add padding for text, graph
    if graph_path is not None:
      graph = Image.open(str(graph_path))
      graph_draw = ImageDraw.Draw(graph)

      # Draw a line on the graph
      x = int(round(graph.size[0]*(this_frame_i/end_frame_i)))
      graph_draw.line((x, 0, x, graph.size[1]), fill=(255,255,255), width=1)
      bottom_padding = graph.size[1]

    rgba_image = Image.new('RGBA', (image.size[0], image.size[1]+top_padding+bottom_padding), (0,0,0,0))
    rgba_image.paste(image, ( 0, top_padding ))
    if graph_path is not None:
      rgba_image.paste(graph, ( 0, top_padding+image.size[1] ))
    image = rgba_image

    # Draw text and progress bar
    font_color = 'rgb(255,255,255)'
    small_font = ImageFont.truetype(str(FONT_PATH), size=14)
    draw = ImageDraw.Draw(image)

    time = frame_idx*frame_rate

    hours = math.floor(time / 3600)
    minutes = math.floor((time - (hours*3600)) / 60)
    seconds = math.floor((time - (hours*3600)) % 60)

    label = "{:02d}h{:02d}'{:02d}\" ({:d})".format(hours, minutes, seconds, frame_idx)
    draw.text((10, 10), label, fill=font_color, font=small_font)

    # Add progress bar
    width = int(frame_idx/len(image_files)*image.size[0])
    draw.rectangle([ (0, 0), (width, 5) ], fill=font_color)

    annotated_frames.append(image)

    # Build tracks if necessary
    if draw_tracks:
      if track_frame is None:
        track_frame = Image.new('RGBA', mask.size, (0,0,0,0))
        track_draw = ImageDraw.Draw(track_frame)

      if old_props is not None:
        old_props.rename(columns={ 'x': 'old_x', 'y': 'old_y' }, inplace=True)
        coords = props.merge(old_props, on='label')
        for row in coords.itertuples():
          track_draw.line([ row.old_x, row.old_y, row.x, row.y ], fill=(row.red, row.green, row.blue), width=2)
      
      old_props = props[['x', 'y', 'label']]

  # Write out movie, adding the track frame if needed
  if scale != 1.0:
    track_frame = track_frame.resize((scale*track_frame.size[0], scale*track_frame.size[1]), resample=Image.BILINEAR)

  fourcc = cv2.VideoWriter_fourcc(*codec)
  writer = cv2.VideoWriter(str(output_file_path), fourcc, 10, (annotated_frames[0].size[0], annotated_frames[0].size[1]), True)
  for annotated_frame in tqdm(annotated_frames, desc='Adding tracks'):
    if draw_tracks:
      annotated_frame.paste(track_frame, ( 0, top_padding ), track_frame)
    annotated_frame = np.asarray(annotated_frame.convert('RGB'))
    writer.write(annotated_frame)
  writer.release()

def _annotate_image(image, f_data):
  draw = ImageDraw.Draw(image)
  font = ImageFont.truetype(str(FONT_PATH), size=16)

  for row in f_data.itertuple():
    particle_id = row.particle_id
    x = row.x_px
    y = row.y_px
    event = row.event if 'event' in row else 'N'

    draw.ellipse([x-CIRCLE_RADIUS, y-CIRCLE_RADIUS, x+CIRCLE_RADIUS, y+CIRCLE_RADIUS], outline=CIRCLE_COLORS[event], width=CIRCLE_THICKNESS[event])
    draw.text((x, y), str(particle_id), font=font, anchor='ms')

  return image
