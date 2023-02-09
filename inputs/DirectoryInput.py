from errors import NoImagesFound
import defusedxml.ElementTree as ET
import tifffile

class DirectoryInput:

  def __init__(self, input_path, data_path, channel):
    self.input_path = input_path
    self.data_path = data_path
    files = list(data_path.glob("*.tif")) + list(data_path.glob("*.TIF")) + list(data_path.glob("*.tiff")) + list(data_path.glob("*.TIFF"))
    files = list(filter(lambda x: x.name[:2] != "._", files))

    if len(files) <= 0:
      raise NoImagesFound

    files.sort(key=lambda x: str(len(str(x))) + str(x).lower())

    self.files = files

  def get_spatial_calibration(self):
    pixel_size = None
    with tifffile.TiffFile(self.files[0]) as tif:
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

    return pixel_size

  def get_files(self, data_set):
    return self.files

  def get_data_sets(self):
    return [ self.input_path.name ]

