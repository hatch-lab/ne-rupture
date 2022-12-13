from errors import NoImagesFound
import re
import numpy as np
from tqdm import tqdm

class NDInput:

  def __init__(self, input_path, data_path, channel):
    self.input_path = input_path
    self.data_path = data_path

    if not data_path.exists():
      raise NoImagesFound

    # Get data_sets (ie, stage positions)
    self.meta = {
      'dims': {},
      'data_sets': {},
      'wavelengths': {},
      'wave_in_filename': False,
      'pixel_size': None
    }
    stage_pattern = re.compile('"Stage[0-9]+"')
    wave_pattern = re.compile('"WaveName[0-9]+"')
    with open(data_path) as f:
      for line in f:
        parts = line.split(",")
        if parts[0] == '"EndFile"\n':
          break
        key = parts[0]
        value = parts[1].strip()
        
        if key == '"DoTimelapse"' and value != "TRUE":
          self.meta['dims']['t'] = 1
        elif key == '"NTimePoints"':
          self.meta['dims']['t'] = int(value)

        if key == '"DoWave"' and value != "TRUE":
          self.meta['dims']['c'] == 1
        elif key == '"NWavelengths"':
          self.meta['dims']['c'] = int(value)

        elif key == '"DoZSeries"' and value != "TRUE":
          self.meta['dims']['z'] = 1
        elif key == '"NZSteps"':
          self.meta['dims']['z'] = int(value)

        elif re.match(stage_pattern, key):
          position = key.replace('"Stage', "")
          position = "s" + position.replace('"', "")
          self.meta['data_sets'][position] = value.replace('"', "")

        elif re.match(wave_pattern, key):
          wavelength = key.replace('"WaveName', "")
          wavelength = 'w' + wavelength.replace('"', "")
          self.meta['wavelengths'][wavelength] = value.replace('"', "")

        elif key == '"WaveInFileName"' and value == "TRUE":
          self.meta['wave_in_filename'] = True

    self.segmentation_channel = 'w' + str(channel)
    if self.meta['wave_in_filename']:
      self.segmentation_channel += self.meta['wavelengths'][self.segmentation_channel]

    self.files = {}
    # Collect all the file names for each data set
    base_name = data_path.name
    search_dir = data_path.parent
    for key, data_set in tqdm(self.meta['data_sets'].items()):
      c_first_search_pattern = base_name + "_" + self.segmentation_channel + "_" + key + "*.TIF"
      s_first_search_pattern = base_name + "_" + key + "_" + self.segmentation_channel + "*.TIF"
      fs = list(search_dir.glob(c_first_search_pattern)) + list(search_dir.glob(s_first_search_pattern))
      fs = list(filter(lambda x: x.name[:2] != "._", fs))
      if self.meta['dims']['t'] > 1:
        fs.sort(key=lambda x: re.sub(r'^.*_t([0-9]+)\.TIF$', r'\g<1>', x))
      self.files[data_set] = fs

    # Get x,y spatial size/calibration
    pixel_size = None
    first_key = self.files.keys()[0]
    with tifffile.TiffFile(self.files[first_key][0]) as tif:
      img = tif.pages[0].asarray()
      self.meta['dims']['x'] = img.shape[1]
      self.meta['dims']['y'] = img.shape[0]
      if 'spatial-calibration-x' in tif.pages[0].description:
        # Try from the description

        metadata = ET.fromstring(tif.pages[0].description)
        plane_data = metadata.find("PlaneInfo")

        for prop in plane_data.findall("prop"):
          if prop.get("id") == "spatial-calibration-x":
            pixel_size= float(prop.get("value"))
            break
      
      elif 'XResolution' in tif.pages[0].tags:
        # Try from the XResolution tag
        pixel_size = tif.pages[0].tags['XResolution'].value

        if len(pixel_size) == 2:
          pixel_size = pixel_size[0]/pixel_size[1]

        pixel_size = 1/pixel_size

    self.meta['pixel_size'] = pixel_size

  def get_files(self, data_set):
    return self.files[data_set]

  def get_data_sets(self):
    return self.files.keys()

  def get_spatial_calibration(self):
    return self.meta['pixel_size']

