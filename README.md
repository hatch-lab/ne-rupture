# Hatch Lab automated NE rupture/repair detection

## Installation

### Requirements
- macOS
- Xcode command line tools
- Homebrew
- Python 3.x
- virtualenv
- Fiji
- R
- git
- GitHub account

### Installation of requirements
If you don’t have the above installed and are on macOS, follow the instructions below.

#### Xcode command line tools
1. Go to https://developer.apple.com/downloads/
2. Create an Apple account if you don’t have one; otherwise, login.
3. Find “Command Line Tools” in the list of downloads.
4. Download the installer.
5. Run the installer (this may take a long time).

#### Homebrew
Homebrew is software that makes it easier to install other programs (like Python).

1. Open Terminal.
2. Copy and paste the following, and hit Return:
  - `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
3. Copy and paste the following, and hit Return:
  - `echo 'PATH="/usr/local/opt/python/libexec/bin:$PATH"' >> ~/.bash_profile`

#### Python 3.7.x
1. In Terminal, copy and paste the following, and hit Return:
  - `brew install python`
2. Once that finishes, check to make sure you have Python 3.7.x installed by copy and pasting the following, then hitting Return:
  - `python --version`
3. You should see "Python 3.7.x" (where x is some number).
4. Finally, install virtualenv by copying and pasting the below, then hitting Return:
  - `pip install --user virtualenv`

#### Fiji
1. Go to: https://imagej.net/Fiji/Downloads, and download Fiji.
2. Make sure to install Fiji to your Applications folder.

#### R
1. Go to: https://cran.r-project.org/bin/macosx, and download the latest package for R.
2. Run the .pkg file.

#### git
1. Copy the line below and hit Return:
  - `brew install git`

#### GitHub
1. Signup for a free GitHub account at http://github.com.
2. Tell an administrator (right now that's Lucian) and they will set you up with the Hatch Lab GitHub repositories.

### Installation of the video analysis tool
1. Make or find a directory where you would like to store all of the files.
2. Everything will be placed inside of a `ne-rupture` folder inside that directory.
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

## Usage
Coming soon.