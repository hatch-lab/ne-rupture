# Hatch Lab automated NE rupture/repair detection

## Installation

### Requirements
- macOS 10.15.x

### Installation of the video analysis tool
1. Install Gurobi
The cell tracking algorithm requires Gurobi, a linear algebra solver. Academic licenses are free, but must be renewed every year.
    1. Create an account at https://www.gurobi.com/
    2. Visit https://www.gurobi.com/downloads/gurobi-software/ and download Gurobi 9.1.*
    3. Get a free academic license key: https://www.gurobi.com/downloads/end-user-license-agreement-academic/
    4. In Terminal (located in /Applications/Utilities), install the license by entering:

  `grbgetkey xxxx-xxxx-xxx`
  where `xxxx-xxxx-xxx` is the license key

2. Make or find a directory where you would like to install the tool in your `Documents` folder.
3. In Terminal (located in /Applications/Utilities), copy and paste the following:
  
  `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/hatch-lab/ne-rupture/stable/install.sh)" && source ~/.zshrc`

4. Follow the prompts.

5. The tool should now be installed.

## Quickstart
Frames are captured and stored as multi-channel, 16-bit TIFF stacks with many images stored in one or more TIFF files. 

To keep things simple, this tool expects files to be organized in a particular way. Each position for each experiment should have its own folder. Within that folder should be an `images/raw` directory that contains those TIFF files: 

```
myexperiment-position-1/
  images/
    raw/
      t1.tiff
      t2.tiff
      t2.tiff
      ...
      t498.tiff
```

Once your files have been organized, copy and paste the following into Terminal:

`ner segment path/to/myexperiment-position-1`

where `path/to/myexperiment-position-1` is the folder you created above.

Once this finishes, run:

`ner annotate path/to/myexperiment-position-1`

to actually annotate your TIFFs.

## Usage
This tool is split into two programs: 
  - one that processes images, segments them, tracks nuclei, and extracts object features
  - a second that classifies each nucleus’s events then outputs statistics and information

### `segment`

The basic usage is:
`ner segment path/to/my/experiment-folder`

This program has a number of options. By default, the first channel of each image is extracted. If you want to extract the second channel, execute:

`ner segment path/to/my/experiment-folder --channel=2`

If you want to use a different tracking gap size, 3 frames instead of the default 5, execute:

`python segment path/to/my/experiment-folder --gap-size=3`

Options can also be combined. The default assumes a 3 min pass time, but if you had a 30 s pass time and wanted to extract the third channel, you would execute:

`ner segment path/to/my/experiment-folder --frame-rate=30 --channel=3`

#### Options
- `--channel` Which channel to extract (defaults to 1)
- `--pixel-size` The pixels / µm. Defaults to 0. If set to 0 and the TIFF files have this information stored in their metadata, it will be extracted automatically.
- `--frame-rate` The pass time, in seconds (defaults to 180 s)
- `--gap-size` When tracking objects, the number of frames where an object can be missing and still be considered the same object (defaults to 5)
- `--roi-size` The size of the search region when tracking objects, as a factor of the median object size (see below for more). Defaults to 2.0
- `--segment-pipeline` The CellProfiler pipeline to use for segmenting objects (defaults to ./preprocessors/cellprofiler/segment.cppipe)
- `--features-pipeline` The CellProfiler pipeline to use for extracting features (defaults to ./preprocessors/cellprofiler/extract.cppipe)
- `--min-track-length`  Any tracks with fewer than these frames will be filtered out; the minimum track length must always be at least 4, in order to generate derivatives (defaults to 5)
- `--edge-filter`  Filters cells that are near the edge of the frame, in pixels (defaults to 50)
- `--accept-tracks`  Whether to just accept the tracks; skip generating a preview (defaults to False)

### `annotate`

The basic usage is:
`ner annotate path/to/my/experiment-folder`

By default, if you quit annotating before going through all of the TIFFs, the next time you run this command, it will pick up where you left off. If you wish to start over instead, pass the `--start-over` flag:

`ner annotate path/to/my/experiment-folder --start-over`

By default, in addition to a CSV file with annotated data, graphs and videos will be generated. This can take a while; if you want to skip making graphs or videos, pass the `--skip-graphs` flag:

`ner annotate path/to/my/experiment-folder --skip-videos`

#### Options
- `--skip-videos` If set, will only output a CSV file; no videos
- `--start-over` Start over with annotation even if you haven’t finished annotating this data set
- `--draw-tracks` If set, the final video of the entire field will be overlaid with object tracks. This is useful for visually tracking cell movement.

## Output
More description coming soon.

## Technical details
The above usage and quickstart describe a simplified interface. The actual program is somewhat more complex.

There are, at present, 2 programs:
- `preprocess.py`, accessed through `ner segment`
- `classify.py`, accessed through `ner annotate`

Each of these programs, in turn, call modules that do most of the heavy lifting. Modules are interchangeable, but should accept the same input and produce the same output.

For example, there are 2 different modules for `preprocess.py`: one that uses CellProfiler for segmentation and feature extraction, and another that uses the Allen Institute’s cell segmenter. The `ner segment` program is actually a short-hand for executing `preprocess.py` with the CellProfiler module. However, you may find that the Allen Institute’s cell segmenter performs better for your specific experiment.

Likewise, `classify.py` has 2 different classifiers available. The first is a “manual” classifier, where you are responsible for marking the start of each event, be it a rupture, mitosis, or apoptosis. This is what is called when `ner annotate` is run.

There is a second classifier, however, that attempts to automatically identify ruptures based on nuclear and cytoplasmic intensities.

Other classifiers could be built, using other features and which could be better suited to specific experiments.

> **Note:** Historically, there was a third program that could learn from the classifications generated with the manual classifier in order to find the best thresholds to use with the automated classifier(s). The whole NE rupture tool underwent some refactoring recently, and this program has not yet been moved over to the new way of doing things.
>
> The envisioned pipeline would be to use the manual classifier on one position; learn from that to optimize the automated classifier; then automatically classify the remaining positions.

### `preprocess.py`
#### Program overview
1. Standardize files:
    - Find all TIFFs in the [input]/images/raw folder (\*.tif, \*.TIF, \*.tiff, \*.TIFF all accepted)

    - Sort the TIFFs by filename, with the assumption that frames will be in alphanumeric order

    - A single TIFF may contain multiple images; standardize files by extracting a single channel from each image and saving this single-channel image to [input]/images/[input]/extracted/[frame-number].tif

    > For example, an 8-frame experiment in `my-experiment-position-1/images/raw/GiantTIFFStack.TIF` would produce:  
    `my-experiment-position-1/images/my-experiment-position-1/extracted/001.tif`,  
    `my-experiment-position-1/images/my-experiment-position-1/extracted/002.tif`,  
    ...,  
    `my-experiment-position-1/images/my-experiment-position-1/extracted/008.tif`

2. Segment:
    - Load the chosen segmentation module, with any module-specific options

    - Ask this segmenter to actually segment the images in [input]/images/[input]/extracted

    - The segmenter will output modified images (eg, gamma corrected, smoothed, etc) to [input]/images/[input] and nucleus segmentation masks to [input]/images/[input]/masks
        - Nucleus masks should be images with a background pixel value = 0 and each individual segment given a pixel value corresponding to its ID number

3. Build tracks:
    - Once the segmenter completes, build tracks, connecting objects across frames using the provided gap size and ROI size options
        - This uses KIT-Sch-GE 2021 Cell Tracker: https://git.scc.kit.edu/KIT-Sch-GE/2021-cell-tracking
        - This tracker will merge and split segmentation masks to align with the assigned tracks, and interpolate masks where none exists between gap frames
  
    - Build MP4 movie of the segmented cells and their tracks

    - Show the user this MP4 and ask if the tracks look satisfactory
        - If no, ask for a new gap size/ROI size input and rebuild tracks
        - If yes, continue

    - Output updated masks to [input]/images/[input]/tracks with each unique tracked object given the same pixel value ID for across all frames

4. Extract features:
    - Ask the segmenter module to extract features given the segmentation masks created by KIT-Sch-GE
        - The segmenter will also identify and save corresponding cytoplasmic regions to [input]/images/[input]/cyto-tracks

5. Filter tracks:
      - If the total length of the data set is more than 8 minutes, any tracks < 8 minutes will be removed
      - Otherwise, any tracks with fewer than 3 frames will be removed

6. Derive features:
      - Take the features extracted by the segmenter module and derive additional features (eg, normalized intensity values, derivatives, etc.)

7. Save
    - Save all features to [input]/input/data.csv

**Example directory structure**

Before `preprocess.py` is run:
```
my-experiment-position-1/
--images/
----raw/
------GiantTIFFStack.TIF
```

After `preprocess.py` is run:
```
my-experiment-position-1/
--images/
----raw/
------GiantTIFFStack.TIF
----my-experiment-position-1/
------0001.tif
------0002.tif
...
------0008.tif
------extracted/
--------0001.tif
--------0002.tif
...
--------0008.tif
------masks/
--------0001.tif
--------0002.tif
...
--------0008.tif
------tracks/
--------0001.tif
--------0002.tif
...
--------0008.tif
------cyto-tracks/
--------0001.tif
--------0002.tif
...
--------0008.tif
--input/
----data.csv
```

#### Basic usage
```
Usage:
  preprocess.py [options] <segmenter> <input> [processor arguments...]

Arguments:
  <segmenter> The kind of image processor to use. Can be cellprofiler or aics
  <input> Path to the directory containing the images/raw directory

Options:
  -h, --help Show this screen.
  --output-dir=<string>  [default: input] The subdirectory to save the resulting CSV file
  --output-name=<string>  [default: data.csv] The name of the resulting CSV file
  --img-dir=<string>  [defaults: INPUT/images/(data_set)] The path to extracted/processed TIFF files
  --data-dir=<string>  Where to find the raw data. Typically determined by the preprocessor you've selected.
  --channel=<int>  [default: 1] The channel to keep (eg, the NLS-3xGFP channel)
  --data-set=<string>  The unique identifier for this data set. If none is supplied, the input directory name will be used.
  --pixel-size=<float>  [default: 0] Pixels per micron. If 0, will attempt to detect automatically from the TIFFs.
  --frame-rate=<int>  [default: 180] The seconds that elapse between frames
  --gap-size=<int>  [default: 5] The maximum gap size when building tracks
  --roi-size=<float>  [default: 2.0] Given a segment at time t, at time t+1, will search an area that is the median shape size*roi_size
  --min-track-length=<int>  [default: 5] Tracks less than this number of frames will be excluded. The minimum track length must be >= 4, so that derivatives can be calculated.
  --edge-filter=<int>  [default: 50] Filters cells out that are within this many pixels of the edge.
  --accept-tracks  [default: False] Whether to just accept the tracks; skip asking to check
```

#### Segmenter requirements
##### Segmentation
`preprocess.py` will provide the segmenter with the extracted images

The segmenter must then in turn save all nucleus masks as unsigned 16-bit grayscale images to [input]/images/[input]/masks. Each image must be of the entire field, with the background set to 0 and the pixel values of each individual mask set to that masks ID.

##### Feature extraction
`preprocess.py` will provide the segmenter with the masks modified by the tracking algorithm

The segmenter must then in turn generate cytoplasmic segments for each given nucleus segment and save these masks in the same format as the nucleus masks to [input]/images/[input]/cyto-tracks

In addition, for each nucleus/cytoplasm pair, the segmenter must return a DataFrame with the following columns:
| Column            | Data type | Description                                                                                     |
| ----------------- | --------- | ----------------------------------------------------------------------------------------------- |
| `frame`           | `int`     | The image frame                                                                                 |
| `particle_id`     | `str`     | The ID of this nucleus; the same as the `mask_id`                                               |
| `mask_id`         | `str`     | The mask ID                                                                                     |
| `x`               | `float`   | The centroid x-position of the nucleus, in µm (if µm/px is known; otherwise, in px)             |
| `x_px`            | `int`     | The centroid x-position of the nucleus, in px (if µm/px is not known, the rounded value of `x`) |
| `y`               | `float`   | The centroid y-position of the nucleus, in µm (if µm/px is known; otherwise, in px)             |
| `y_px`            | `int`     | The centroid y-position of the nucleus, in px (if µm/px is not known, the rounded value of `y`) |
| `area`            | `float`   | The area of the nucleus, in µm^2                                                                |
| `mean`            | `float`   | The mean intensity of the nucleus                                                               |
| `median`          | `float`   | The median intensity of the nucleus                                                             |
| `min`             | `float`   | The minimum intensity of the nucleus                                                            |
| `max`             | `float`   | The maximum intensity of the nucleus                                                            |
| `sum`             | `float`   | The total intensity of all nucleus pixels                                                       |
| `cyto_area`       | `float`   | The area of the cytoplasm segment, in µm^2                                                      |
| `cyto_mean`       | `float`   | See the nucleus equivalent                                                                      |
| `cyto_median`     | `float`   | See the nucleus equivalent                                                                      |
| `cyto_min`        | `float`   | See the nucleus equivalent                                                                      |
| `cyto_max`        | `float`   | See the nucleus equivalent                                                                      |
| `cyto_sum`        | `float`   | See the nucleus equivalent                                                                      |
| `x_conversion`    | `float`   | How µm are converted to px in the x dimension (this should be the same as `y_conversion`)       |
| `y_conversion`    | `float`   | How µm are converted to px in the y-dimension (this should be the same as `x_conversion`)       |
| `unit_conversion` | `str`     | If we don’t actually know µm/px, this will be set to `px`; otherwise, `um/px`                   |

#### Output
The final output of `preprocess.py` are images and a CSV file:

*Images*
| Location                             | Description                                                                                                                            |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `[input]/images/[input]/extracted`   | Individual, single-channel copies extracted from the raw images                                                                        |
| `[input]/images/[input]`             | Modified images generated by the segmentation module; often gamma-corrected, smoothed, etc.                                            |
| `[input]/images/[input]/masks`       | Nucleus segmentation masks generated by the segmentation module                                                                        |
| `[input]/images/[input]/tracks`      | Nucleus segmentation masks modified by the cell tracker to align masks with predicted cell tracks                                      |
| `[input]/images/[input]/cyto-tracks` | Cytoplasm segmentation masks derived by the segmentation module from the nucleus segmentation masks in `[input]/images/[input]/tracks` |

*CSV file*
| Column            | Data type | Description                                                                                     |
| ----------------- | --------- | ----------------------------------------------------------------------------------------------- |
| `frame`           | `int`     | The image frame                                                                                 |
| `particle_id`     | `str`     | The ID of this nucleus; the same as the `mask_id`                                               |
| `mask_id`         | `str`     | The mask ID                                                                                     |
| `x`               | `float`   | The centroid x-position of the nucleus, in µm (if µm/px is known; otherwise, in px)             |
| `x_px`            | `int`     | The centroid x-position of the nucleus, in px (if µm/px is not known, the rounded value of `x`) |
| `y`               | `float`   | The centroid y-position of the nucleus, in µm (if µm/px is known; otherwise, in px)             |
| `y_px`            | `int`     | The centroid y-position of the nucleus, in px (if µm/px is not known, the rounded value of `y`) |
| `area`            | `float`   | The area of the nucleus, in µm^2                                                                |
| `mean`            | `float`   | The mean intensity of the nucleus                                                               |
| `median`          | `float`   | The median intensity of the nucleus                                                             |
| `min`             | `float`   | The minimum intensity of the nucleus                                                            |
| `max`             | `float`   | The maximum intensity of the nucleus                                                            |
| `sum`             | `float`   | The total intensity of all nucleus pixels                                                       |
| `cyto_area`       | `float`   | The area of the cytoplasm segment, in µm^2                                                      |
| `cyto_mean`       | `float`   | See the nucleus equivalent                                                                      |
| `cyto_median`     | `float`   | See the nucleus equivalent                                                                      |
| `cyto_min`        | `float`   | See the nucleus equivalent                                                                      |
| `cyto_max`        | `float`   | See the nucleus equivalent                                                                      |
| `cyto_sum`        | `float`   | See the nucleus equivalent                                                                      |
| `x_conversion`    | `float`   | How µm are converted to px in the x dimension (this should be the same as `y_conversion`)       |
| `y_conversion`    | `float`   | How µm are converted to px in the y-dimension (this should be the same as `x_conversion`)       |
| `unit_conversion` | `str`     | If we don’t actually know µm/px, this will be set to `px`; otherwise, `um/px`                   |
| `frame_rate`      | `int`     | The seconds between each frame |
| `time`            | `int`     | `frame`×`frame_rate` |
| `data_set`        | `str`     | The input folder name |
| `normalized_median` | `float` | Given cell *i*, ![median-avg(median)](http://www.sciweavers.org/tex2img.php?eq=median_%7Bi%7D-%5Coverline%7Bmedian%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) |
| `normalized_mean`   | `float` | Given cell *i*, ![mean-avg(median)](http://www.sciweavers.org/tex2img.php?eq=mean_%7Bi%7D-%5Coverline%7Bmedian%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) |
| `normalized_sum`    | `float` | Given cell *i*, ![sum-avg(median)](http://www.sciweavers.org/tex2img.php?eq=sum_%7Bi%7D-%5Coverline%7Bmedian%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) |
| `normalized_cyto_mean` | `float` | Given cell *i*, ![cyto_mean-avg(median)](http://www.sciweavers.org/tex2img.php?eq=cyto_mean_%7Bi%7D-%5Coverline%7Bmedian%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) |
| `scaled_area`       | `float` | Given cell *i*, `area` scaled between 0–1 |
| `scaled_cyto_area`  | `float` | Given cell *i*, `cyto_area` scaled between 0–1 |
| `stationary_area`   | `float` | Apply a 1st order difference to `scaled_area`, using a sliding window average of 1 hr with a 30 min step size |
| `stationary_cyto_area` | `float` | Apply a 1st order difference to `scaled_cyto_area`, using a sliding window average of 1 hr with a 30 min step size |
| `stationary_median` | `float` |  Apply a 1st order difference to `normalized_median`, using a sliding window average of 1 hr with a 30 min step size |
| `stationary_mean`   | `float` |  Apply a 1st order difference to `normalized_mean`, using a sliding window average of 1 hr with a 30 min step size |
| `stationary_sum`    | `float` |  Apply a 1st order difference to `normalized_sum`, using a sliding window average of 1 hr with a 30 min step size |
| `stationary_cyto_mean` | `float` |  Apply a 1st order difference to `normalized_cyto_mean`, using a sliding window average of 1 hr with a 30 min step size |
| `area_spline`       | `float` | A cubic spline of `stationary_area` |
| `area_derivative`   | `float` | The 1st derivative of `area_spline` as a function of `time` (ie, the change in area over time) |
| `cyto_area_spline`  | `float` | A cubic spline of `stationary_cyto_area` |
| `cyto_area_derivative` | `float` | The 1st derivative of `cyto_area_spline` as a function of `time` (ie, the change in area over time) |
| `median_spline`     | `float` | A cubic spline of `stationary_median` |
| `median_derivative` | `float` | The 1st derivative of `median_spline` as a function of `time` (ie, the change in median intensity over time) |
| `mean_spline`       | `float` | A cubic spline of `stationary_mean` |
| `mean_derivative`   | `float` | The 1st derivative of `mean_spline` as a function of `time` (ie, the change in mean intensity over time) |
| `cyto_mean_spline`  | `float` | A cubic spline of `stationary_cyto_mean` |
| `cyto_mean_derivative` | `float` | The 1st derivative of `cyto_mean_spline` as a function of `time` (ie, the change in cytoplasmic mean intensity over time) |
| `sum_spline`        | `float` | A cubic spline of `stationary_sum` |
| `sum_derivative`    | `float` | The 1st derivative of `sum_spline` as a function of `time` (ie, the change in total nuclear intensity over time) |
| `x_spline`          | `float` | A cubic spline of `x` |
| `x_derivative`      | `float` | The 1st derivative of `x_spline` as a function of `time` (ie, the x velocity) |
| `y_spline`          | `float` | A cubic spline of `y` |
| `y_derivative`      | `float` | The 1st derivative of `y_spline` as a function of `time` (ie, the y velocity) |
| `speed`             | `float` | The total magnitude of `x_derivative` and `y_derivative` (ie, the total speed, without regard to direction) |
| `nearest_neighbor`  | `str`   | The ID of the nearest nucleus mask at a given frame |
| `nearest_neighbor_distance` | `float` | The distance to the nearest neighbor |


#### Available segmenters
##### `cellprofiler`
This will use CellProfiler to segment nuclei, cytoplasms, and extract features.

It comes with two additional options for specifying the pipelines to use. Because these pipelines have special requirements, if you wish to use a different pipeline, best practice is to copy the default pipeline and modify it.

###### `--segment-pipeline`
The `.cppipe` file to use for segmentation.  
Defaults to a pipeline that comes with the NE rupture tool, located in `./preprocessors/cellprofiler/segment.cppipe`.

**Necessary modules**  
*SaveImages*  
This pipeline must have a `SaveImages` module for the processed images (whether they be rescaled, smoothed, etc.) with the following settings:

| Setting | Value |
| ------- | ----- |
| Type of image to save | Image |
| Method for constructing file names | From image filename |
| Image name for file prefix | Orig |
| Append a suffix to the image file name | No |
| Saved file format | tiff |
| Image bit depth | 8-bit integer |
| Save with lossless compression | Yes |
| Output file location | *This setting’s value will be ignored* |
| Overwrite existing files without warning | Yes |
| When to save | Every cycle |
| Record the file and path information | No |
| Create subfolders | No |

*ConvertObjectsToImage*  
This module must be used to convert the nucleus mask objects to images, with the following settings:

| Setting | Value |
| ------- | ----- |
| Color format | uint16 |

*SaveImages*  
This pipeline must have a second `SaveImages` module for the masks (converted from objects to images by the `ConvertObjectsToImage` module). This must have the following settings:

| Setting | Value |
| ------- | ----- |
| Type of image to save | Image |
| Method for constructing file names | From image filename |
| Image name for file prefix | Orig |
| Append a suffix to the image file name | Yes |
| Text to append to the image name | `-mask` |
| Saved file format | tiff |
| Image bit depth | 16-bit integer |
| Save with lossless compression | Yes |
| Output file location | *This setting’s value will be ignored* |
| Overwrite existing files without warning | Yes |
| When to save | Every cycle |
| Record the file and path information | No |
| Create subfolders | No |


##### `--features-pipeline`
The `.cppipe` file to use for cytoplasmic segmentation and feature extraction.  
Defaults to a pipeline that comes with the NE rupture tool, located in `./preprocessors/cellprofiler/extract.cppipe`.

**Necessary modules**  
*NamesAndTypes*  
This module must be configured with two criteria:

| Setting | Value |
| ------- | ----- |
| Assign a name to | Images matching rules |
| Process as 3D | No |
|  |  |
| Select the rule criteria | Match `All` of the following rules: `Directory` `Does` `Contain` `tracks` |
| Name to assign these images | `masks` |
| Select the image type | Grayscale |
| Set intensity range from | Image bit-depth |
|  |  |
| Select the rule criteria | Match `All` of the following rules: `Directory` `Does not` `Contain` `tracks` |
| Name to assign these images | `Orig` |
| Select the image type | Grayscale |
| Set intensity range from | Image metadata |

*ConvertImageToObjects*  
This module must be used to convert the nucleus mask images to objects, with the following settings:

| Setting | Value |
| Select the input image | masks |
| Name the output object | *whatever is convenient* |
| Convert to boolean image | No |
| Preserve original labels | Yes |

*IdentifySecondaryObjects*  
*IdentifyTertiaryObjects*  
Configure CellProfiler to identify the cytoplasm from the nuclear masks as you see fit.

*MeasureObjectIntensity*  
*MeasureObjectSizeShape*  
Configure these modules to record the intensities and size/shapes of both the nuclei and cytoplasm objects.

*ConvertObjectsToImage*  
This module must be used to convert the cytoplasm mask objects to images, with the following settings:

| Setting | Value |
| ------- | ----- |
| Color format | uint16 |

*SaveImages*  
This pipeline must have a second `SaveImages` module for the cytoplasm masks (converted from objects to images by the `ConvertObjectsToImage` module). This must have the following settings:

| Setting | Value |
| ------- | ----- |
| Type of image to save | Image |
| Method for constructing file names | From image filename |
| Image name for file prefix | Orig |
| Append a suffix to the image file name | Yes |
| Text to append to the image name | `-cyto-mask` |
| Saved file format | tiff |
| Image bit depth | 16-bit integer |
| Save with lossless compression | Yes |
| Output file location | *This setting’s value will be ignored* |
| Overwrite existing files without warning | Yes |
| When to save | Every cycle |
| Record the file and path information | No |
| Create subfolders | No |

*ExportToSpreadsheet*  
This pipeline must output measured values with the following settings:

| Setting | Value |
| ------- | ----- |
| Select the column delimiter | Comma (“,”) |
| Output file location | *This setting’s value will be ignored* |
| Add a prefix to file names | No |
| Overwrite existing files without warning | Yes |
| Add image metadata columns to your object data file | No |
| Add image file and folder names to your object data file | Yes |
| Representation of NaN/Inf | NaN |
| Select the measurements to export | Yes |
| Calculate the per-image mean values | No |
| Calculate the per-image median values | No |
| Calculate the per-image standard deviation values | No |
| Export all measurement types | No |
| Data to export | *The nuclear objects* |
| Use the object name for the file name | No |
| File name | ShrunkenNuclei.csv |
| Data to export | *The cytoplasm objects* |
| Combine these object measurements with those of the previous object | Yes |

Measurements to export must include:

- Cytoplasm / AreaShape / Area
- Cytoplasm / Intensity / IntegratedDensity / Orig
- Cytoplasm / Intensity / MeanIntensity / Orig
- Cytoplasm / Intensity / MedianIntensity / Orig
- Cytoplasm / Intensity / MaxIntensity / Orig
- Cytoplasm / Intensity / MinIntensity / Orig
- Nuclei / AreaShape / Area
- Nuclei / AreaShape / Center
- Nuclei / Intensity / IntegratedDensity / Orig
- Nuclei / Intensity / MeanIntensity / Orig
- Nuclei / Intensity / MedianIntensity / Orig
- Nuclei / Intensity / MaxIntensity / Orig
- Nuclei / Intensity / MinIntensity / Orig

Any other measurements will be ignored.

##### `aics`
This will use the Allen Institute’s cell segmenter to segment nuclei, cytoplasms, and extract features.

It comes with three additional options:

`--gamma=<float>`: The gamma correction to use (defaults to 0.5)
`--rolling-ball-size`: The rolling ball diameter to use for rolling ball subtraction, in um (defaults to 100)
`--mip-dir`: The path to export maximum intensity projections (defaults: [input]/images/[input]/mip)

### `classify.py`
#### Program overview
1. Load [input]/input/data.csv

2. Classify events
    - Load the chosen classification module, with any module-specific options

    - Ask the classifier to actually classify events

    - The classifier will return with classified data

5. Save updated data to [input]/output/results.csv

6. Save cell summary and event summary to [input]/output/cell-summary.csv and [input]/output/event-summary.csv, respectively.

7. Generate videos, stored in [input]/output/videos/
    - After classification, a video of the entire field is saved to [input]/output/videos/[input].mp4. This field will feature a circle around each segmented cell and its ID. During events, the circle will be colored such that ruptures are red, repairs green, mitosis blue, and apoptosis gray.
    - Additional videos are made for each cell, cropped around that cell and with the corresponding `stationary_mean` and `stationary_cyto_mean` drawn below it. Any rupture/repair events are annotated on that graph as a shaded region.

**Example directory structure**

Before `classify.py` is run:
```
my-experiment-position-1/
--images/
----..abridged for clarity..
--input/
----data.csv
```

After `classify.py` is run:
```
my-experiment-position-1/
--images/
----..abridged for clarity..
--input/
----data.csv
--output/
----results.csv
----event-summary.csv
----cell-summary.csv
----videos/
------[input].mp4
------[input]/
--------1.mp4
--------2.mp4
...
--------[n].mp4
```

#### Basic usage
```
Usage:
  classify.py [options] <classifier> <input> [classifier arguments...]

Arguments:
  <classifier> The name of the classifier to use
  <input> Path to the directory containing the images/ and input/ directories

Options:
  -h, --help Show this screen.
  --input-name=<string>  [default: data.csv] The name of the input CSV file
  --output-name=<string>  [default: results.csv] The name of the resulting CSV file
  --cell-summary-name=<string>  [default: cell-summary.csv] The name of the CSV file for data summarized by cells
  --event-summary-name=<string>  [default: event-summary.csv] The name of the CSV file for data summarized by events
  --output-dir=<string>  [default: output] The name of the subdirectory in which to store output
  --input-dir=<string>  [default: input] The name of the subdirectory in which to find the inpute CSV file
  --skip-videos  Whether to skip producing videos
  --img-dir=<string>  [default: images] The subdirectory that contains TIFF images of each frame, for outputting videos.
  --conf=<string>  Override configuration options in conf.json with a JSON string.
  --draw-tracks  [Default: False] Whether to overlay tracks on the annotated movies
  ```
#### Classifier requirements
The classifier must be able to process a CSV file with the columns specified above in the segmenter output.

It must in turn output a DataFrame with the same columns, along with the following, additional columns:

| Column            | Data type | Description                                                                                     |
| ----------------- | --------- | ------------------------------------------------------------------------------------------------|
| `event`           | `str`     | The event occurring for this cell at this frame. This can be `N`, `R`, `E`, `X`, or `M`. |
| `event_id`        | `int`     | An ID for this event, unique to this specific cell (ie, two different cells may both have events with an ID of 1). This is `-1` for all `N` event frames and is 0-indexed (the first event ID is 0). |
| `repair_k`        | `float`   | All repair events will be fitted to a one-phase curve in order to estimate hole size. The resulting rate constant `k` will be saved here, or will be NaN for all non-repair events or if a curve could not be fit. |
| `event_duration`  | `int`     | The total duration of this event, in seconds. |

The `event` column can be one of the following single-letter codes:  
**N:** No event  
**R:** Rupture  
**E:** Repair  
**X:** Apoptosis  
**M:** Mitosis  

Each event will be given its own event ID, with the first event for a given cell given the ID of 0. 

The exception to this is for rupture/repair events. Each rupture and its corresponding repair will have the same event ID.

#### Available classifiers
##### `manual`
This will open a GUI for stepping through each segmented cell and specifying the **start** of any event. All that is required is for you to mark the first frame an event starts. The classifier will then determine the duration of that event. This is termed *event seeding.*

Event seeds are extended as follows:
1. A baseline `stationary_median` is calculated by taking the mean of all frames that are marked as **N** (for no event).
2. From all given seed points, extend each event forwards and backwards in time until one of the following conditions is met:
    - We reach the beginning/end of the track
    - The `stationary_median` value reaches at least 92% of the baseline
    - We reach a frame that is not a non-event frame
3. Since we have converted some non-event frames into event-frames, this will affect the baseline `stationary_median` value; recalculate this value. If the squared difference between the old and new baseline is >= 3E-5, extend events again. Otherwise, stop.

For all rupture events, we then identify the beginning of repair by generating a smoothed average of the `stationary_median` value, using a sliding window average with a window size of ~13 min and a step size of ~4 min. The first frame where this smoothed median begins returning to baseline is marked as the beginning of repair, and the remaining frames of this event are considered repair.

Thereafter, this classifier will attempt to fit a one-phase association curve to the `stationary_median` value, storing the fitted rate constant k in the `repair_k` column. If no curve can be fit, the column’s value will be NaN.

This classifier has three additional options:
###### `--distance-filter=0`
Filter out any cells whose nearest neighbor is at any time less than the supplied value. Defaults to 0, for no filtering.

###### `--jumpy-filter=0.0`
Filter out any cells whose speed is greater than the supplied value, in µm/frame. Defaults to 0, for no filtering.  
This can be useful for filtering out segmentation errors. When two cells are very close, their tracks can split and merge repeatedly. The effect of this is to artificially cause the cell’s speed to be very high during the split/merge events, since the centroid positions will abruptly shift from the center of one nucleus to the average center of two.

###### `--start-over`
Because this classifier requires hand annotation, this can take a long time. The classifier, by default, will save as you progress through a given [input]. In the event your computer crashes (or you quit before reaching the end), the next time you run it, it will start where you left off.

However, if you want to start over, pass this flag and you will start at the beginning.

##### `fixed-cutoff`
This is a truly automated classifier, and uses thresholding to determine event seeds. 

Any frames where the `stationary_mean` is less than some value **and** the `stationary_cyto_mean` is greater than some value will be marked as ruptures. The logic is that, during ruptures, nucleoplasmic signal will diffuse into the cytoplasm, dropping the average nuclear signal while raising the average cytoplasmic signal.

From there, seeded events are extended in a similar fashion as above, with one exception. During event extension, the only criteria to stop extension are:
    - We reach the beginning/end of the track
    - The `stationary_median` value reaches at least 92% of the baseline

If we run into frames already marked as ruptures by the initial thresholding, these frames will be merged into the same extended event. This may result in rapid sequential ruptures being classified as a single rupture.

This classifier has no other options. At present, threshold values cannot be adjusted from the command line, but are stored in a configuration file located in ./classifiers/fixed-cutoff/conf.json.

### Examples
Options for the general program must precede options for a specific module.

Replicating `ner segment ../my-experiment-position-1`:  
`python preprocess.py cellprofiler ../my-experiment-position-1`

Specify an alternative segmentation pipeline:  
`python preprocess.py cellprofiler ../my-experiment-position-1 --segment-pipeline=../path/to/another/pipeline.cppipe`

Specify an alternative pass-time of 1 minute with an alternative segmentation pipeline:  
`python preprocess.py --frame-rate=60 cellprofiler ../my-experiment-position-1 --segment-pipeline=../path/to/another/pipeline.cppipe`
> Note how `--frame-rate` comes after `preprocess.py` and `--segment-pipeline` after `cellprofiler`/

Use AICS instead of CellProfiler with a gamma correction of 0.6, extracting the 3rd channel of each image:  
`python preprocess.py --channel=3 aics ../my-experiment-position-1 --gamma=0.6`

Replicating `ner annotate ../my-experiment-position-1`:  
`python classify.py manual ../my-experiment-position-1`

Specify a distance filter:  
`python classify.py manual ../my-experiment-position-1 --distance-filter=4`

Start annotation over:  
`python classify.py manual ../my-experiment-position-1 --start-over`

Run the `fixed-cutoff` classifier, drawing tracks over the produced movie:  
`python classify.py --draw-track fixed-cutoff ../my-experiment-position-1`
