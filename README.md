# Hatch Lab automated NE rupture/repair detection

## Basic workflow
1. Frames are captured and stored as RGB, 16-bit .tif files
2. All frames will be converted to a single TIFF stack
3. Stretch instensity over entire 16-bit histogram
  - This linearly maps intensity values such that min(intensity) -> 0, max(intensity) -> 2^16
4. Convert to 8-bit grayscale
5. Apply 5 px median filter
6. Apply gamma correction of 0.50
7. Throw out all channels except GFP (channel 2)
8. For each frames, subtract the mean frame intensity
9. Import into Imaris
10. Threshold using specified threshold, watershed params
  - Requires experiment-specific parameters
11. Perform motion tracking on identified surfaces using autoregressive motion tracking and the supplied params
  - Requires experiment-specific parameters
12. Export position, mean intensity, and area for each nucleus/time-point
13. Identify rupture events, repair events, cell-death events
14. Export interesting statistics
  - # nuclei that have experienced >= 1 rupture/all nuclei
  - For each nucleus, # rupture events, # of repair events, duration of rupture events, rate of GFP-intensity recovery following repair
15. Export QA video annotating track IDs and events

## Software requirements
- Python
- Imaris
- FIJI
- ?

## Installation

## Usage