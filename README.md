# Hatch Lab automated NE rupture/repair detection

## Installation

### Requirements
- macOS
- Xcode command line tools
- Homebrew
- Python 3.x
- virtualenv
- MATLAB
- R
- git

### Installation of requirements
If you don’t have the above installed and are on macOS, follow the instructions below.

#### Xcode command line tools
*Necessary for installation of Python.*

1. Go to https://developer.apple.com/downloads/
2. Create an Apple account if you don’t have one; otherwise, login.
3. Find “Command Line Tools” in the list of downloads.
4. Download the installer.
5. Run the installer (this may take a long time).

#### Homebrew
*Homebrew is software that makes it easier to install other programs (like Python).*

1. Open Terminal.
2. Copy and paste the following, and hit Return:
  - `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
3. Copy and paste the following, and hit Return:
  - `echo 'PATH="/usr/local/opt/python/libexec/bin:$PATH"' >> ~/.bash_profile`

#### Python 3.7.x
*The latest version of Python*

1. In Terminal, copy and paste the following, and hit Return:
  - `brew install python`
2. Once that finishes, check to make sure you have Python 3.7.x installed by copy and pasting the following, then hitting Return:
  - `python --version`
3. You should see "Python 3.7.x" (where x is some number).
4. Finally, install virtualenv by copying and pasting the below, then hitting Return:
  - `pip install --user virtualenv`

#### MATLAB
*Used for segmentating cells and extracting data.*

1. Buy MATLAB and follow instructions for installation from https://www.mathworks.com/products/matlab.html.

#### R
*Used for making pretty graphs.*

1. Go to: https://cran.r-project.org/bin/macosx, and download the latest package for R.
2. Run the .pkg file.

#### git
*Used for keeping up-to-date with all of Hatch Lab’s tools.*

1. Copy the line below and hit Return:
  - `brew install git`

### Installation of the video analysis tool
1. Make or find a directory where you would like to install the tool—it will be placed inside of a `ne-rupture` folder inside that directory.
3. In Terminal, type the following and hit Return:
  - `cd "~/path/to/my/folder"`
  - For example, if I want to install into `Documents/Hatch Lab`, I would type `cd "~/Documents/Hatch Lab"`
4. Retrieve the latest version of the analysis tool by copying and pasting the following, then hitting Return:
  - `git clone --branch stable https://github.com/hatch-lab/ne-rupture.git`
5. Copy and paste each line below and hit Return:
  - `virtualenv ne-rupture`
  - `cd ne-rupture`
  - `source bin/activate`
  - `pip install -r requirements.txt`
6. The tool should now be installed.

## Basic workflow
Frames are captured and stored as multi-channel, 16-bit TIFF stacks with many images stored in one or more TIFF files. We need to convert these into individual 8-bit grayscale images, and processed to make segmentation easier.

The program `preprocess.py` will, for each image:
  - Extract the desired channel
  - Stretch the image intensity over a 16-bit range
  - Convert to 8-bit grayscale
  - Perform median filtering (defaults to 8 px)
  - Perform gamma correction (defaults to 0.5)
  - Perform 30 µm median filter background subtraction (similar to a rolling-ball background subtraction)
  - Save the processed image to a specified directory

Then, it will use MATLAB to find cells, extract features, and write out a CSV file with everything we’d ever want.

The program `classify.py` will have you annotate each cell, marking whenever there is a rupture, mitosis, or apoptosis. (Alternatively, you can have it try to do this for you automatically.) Following, it will generate graphs, annotated movies, and statistics.

The basic workflow is to run `preprocess.py` followed by `classify.py`.

## Basic file organization
To keep things simple, this tool expects files to be organized in a particular way. 

Create a folder to hold the data for a particular image sequence you want to analyze. Give it a unique name. This name will be used to identify the data from this image sequence.

For example, I might create the following folder:

`~/Documents/experiments/2018-08-LD-rotation-well-1`

My data would then be identified as coming from a `2018-08-LD-rotation-well-1` data set.

Within that folder, create an `images` folder and within that, a `raw` folder. Copy your TIFFs to the `raw` folder. For example, I would copy my TIFFs to

`~/Documents/experiments/2018-08-LD-rotation-well-1/images/raw`

After running `preprocess.py`, some additional folders and files will have been created. You should see:
`~/Documents/experiments/2018-08-LD-rotation-well-1/images/2018-08-LD-rotation-well-1`, containing the processed TIFFs for each frame, as well as `~/Documents/experiments/2018-08-LD-rotation-well-1/input/data.csv`, a CSV file of data for each cell in the video.

After running `classify.py`, you should have a `~/Documents/experiments/2018-08-LD-rotation-well-1/output` folder with graphs and videos and a `results.csv` file containing information about each cell in the video, including rupture events.

## Usage
### `preprocess.py`

The basic usage is:
`python preprocess.py matlab path/to/my/experiment-folder`

This program has a number of options. By default, the first channel of each image is extracted. If I wanted to extract the second channel, I would execute:

`python preprocess.py matlab path/to/my/experiment-folder --channel=2`

If I wanted to use a different median filter window size, 10 px instead of the default of 8 px, I would execute:

`python preprocess.py matlab path/to/my/experiment-folder --filter-window=10`

Options can also be combined. The default assumes a 3 min pass time, but if you had a 30 s pass time and wanted to extract the third channel, you would execute:

`python preprocess.py matlab path/to/my/experiment-folder --frame-rate=30 --channel=3`

#### Options
`--channel` Which channel to extract (defaults to 1)
`--filter-window` The window radius for median filtering, in pixels (defaults to 8 px)
`--gamma` The gamma correction value to use (defaults to 0.5)
`--pixel-size` The pixels / µm. Defaults to 1. If set to 0 and the TIFF files have this information stored in their metadata, it will be extracted automatically.
`--rolling-ball-size` The disc radius to use for background subtraction, in microns; this is similar to a rolling ball radius (defaults to 100 µm)
`--frame-rate` The pass time, in seconds (defaults to 180 s)

### `classify.py`

The basic usage is:
`python classify.py manual path/to/my/experiment-folder`

There are 2 classifiers that you can choose from: `manual` and `fixed-cutoff`. The second tries to automatically identify ruptures, to mixed success. In general, you should use `manual`, which will have you annotate when ruptures occur.

If you want to try the automatic classifier, run:
`python classify.py fixed-cutoff path/to/my/experiment-folder`

If you don’t need graphs or videos, you can run:
`python classify.py manual path/to/my/experiment-folder --skip-graphs`

This will only output the CSV file.