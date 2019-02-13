# Hatch Lab automated NE rupture/repair detection

## Basic workflow
1. Frames are captured and stored as RGB, 16-bit .tif files
2. imaris-preprocessor.py will use Fiji to run the following on each image:
  - Split out GFP channel
  - Stretch intensity over 16-bit range
  - Convert 8-bit grayscale
  - Perform median filtering (defaults to 5 px)
  - Perform gamma correction (defaults to 0.5)
  - Perform 30 µm rolling-ball background subtraction
  - Calibrate pixel sizes (20x SD defaults to 0.5089)
  - Save the processed image to the specified directory
  - Save a TIFF stack of all images to the specified directory
3. Import TIFF stack into Imaris:
  - Threshold using desired threshold, watershed parameters
4. Export necessary parameters for a desired classifier
  - Basic:
    Needs position, area, intensity median, and intensity sum
  - Outliers:
    Needs position, area, and intensity median
5. Run the desired classifier
  - CSV of statistics output to desired directory
6. Optionally, generate graphs and videos of identified cells

## Software requirements
- See: https://3.basecamp.com/3291612/buckets/9521299/documents/1581931424

## Installation
- See https://3.basecamp.com/3291612/buckets/9521299/documents/1581931424

## Usage
Coming soon.