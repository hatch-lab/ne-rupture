TODO:
Go through packages and install one by one to get updated requirements.txt
Go through and upgrade packages as needed

# Hatch Lab automated NE rupture/repair detection

## Installation

### Requirements
- macOS 10.15.x

### Installation of the video analysis tool
1. Make or find a directory where you would like to install the tool in your `Documents` folder.
2. In Terminal (located in /Applications/Utilities), copy and paste the following:

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/hatch-lab/ne-rupture/stable/install.sh)"`
3. The tool should now be installed.

## Quickstart
Frames are captured and stored as multi-channel, 16-bit TIFF stacks with many images stored in one or more TIFF files. 

To keep things simple, this tool expects files to be organized in a particular way. Each position for each experiment should have its own folder. Within that folder should be a `images/raw` directory that contains those TIFF files: 

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
  - one that processes images, segments them, and identifies the nuclei
  - a second that classifies each nucleus’s events then outputs statistics and information

### `segment`

The basic usage is:
`ner segment path/to/my/experiment-folder`

This program has a number of options. By default, the first channel of each image is extracted. If you want to extract the second channel, execute:

`ner segment path/to/my/experiment-folder --channel=2`

If you want to use a different median filter window size, 10 px instead of the default of 8 px, execute:

`python preprocess.py matlab path/to/my/experiment-folder --filter-window=10`

Options can also be combined. The default assumes a 3 min pass time, but if you had a 30 s pass time and wanted to extract the third channel, you would execute:

`ner segment path/to/my/experiment-folder --frame-rate=30 --channel=3`

#### Options
- `--channel` Which channel to extract (defaults to 1)
- `--filter-window` The window radius for median filtering, in pixels (defaults to 8 px)
- `--gamma` The gamma correction value to use (defaults to 0.5)
- `--pixel-size` The pixels / µm. Defaults to 0. If set to 0 and the TIFF files have this information stored in their metadata, it will be extracted automatically.
- `--rolling-ball-size` The disc radius to use for background subtraction, in microns; this is similar to a rolling ball radius (defaults to 100 µm)
- `--frame-rate` The pass time, in seconds (defaults to 180 s)

### `annotate`

The basic usage is:
`ner annotate path/to/my/experiment-folder`

By default, if you quit annotating before going through all of the TIFFs, the next tiume you run this command, it will pick up where you left off. If you wish to skip this, pass the `--start-over` flag:

`ner annotate path/to/my/experiment-folder --start-over`

By default, in addition to a CSV file with annotated data, graphs and videos will be generated. This can take a while; if you want to skip making graphs or videos, pass the `--skip-graphs` flag:

`ner annotate path/to/my/experiment-folder --start-over`

#### Options
- `--skip-graphs` If set, will only output a CSV file; no graphs or videos
- `--start-over` Start over with annotation even if you haven’t finished annotating this data set