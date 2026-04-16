# Flir Image Extractor

FLIR® thermal cameras like the FLIR ONE® include both a thermal and a visual light camera.
The latter is used to enhance the thermal image using an edge detector.

The resulting image is saved as a jpg image but both the original visual image and the raw thermal sensor data are embedded in the jpg metadata.

This small Python tool/library allows to extract the original photo and thermal sensor values converted to temperatures.

This fork also adds:

- selectable colormaps from the command line
- a separate `colormaps.py` registry for custom and built-in maps
- interactive plot controls for markers, reference markers, delta markers, deletion, and display sliders

## Requirements

This tool relies on `exiftool`. It should be available in most Linux distributions (e.g. as `perl-image-exiftool` in Arch Linux or `libimage-exiftool-perl` in Debian and Ubuntu).

It also needs the Python packages *numpy*, *matplotlib*, *pillow*, *scipy*, *cmocean*, *mplcursors*, and *scienceplots*.

```bash
# sudo apt update
# sudo apt install exiftool python-setuptools
# sudo pip install numpy matplotlib pillow scipy cmocean mplcursors scienceplots
```

## Usage

This module can be used by importing it:

```python
import flir_image_extractor
fir = flir_image_extractor.FlirImageExtractor()
fir.process_image('examples/ax8.jpg')
fir.plot()
```

Or by calling it as a script:

```bash
python flir_image_extractor.py -p -i 'examples/zenmuse_xtr.jpg'
```

Using a specific colormap:

```bash
python flir_image_extractor.py -p -i 'examples/zenmuse_xtr.jpg' -c flir_high_contrast
```

Exporting CSV while also saving the rendered thermal image:

```bash
python flir_image_extractor.py -i 'examples/zenmuse_xtr.jpg' -csv thermal.csv -c fluke
```

```bash
usage: flir_image_extractor.py [-h] -i INPUT [-p]
                               [-c {flir,flir_high_contrast,fluke,fluke_high_contrast,hot,inferno,oxy,plasma,seismic,thermal}]
                               [-exif EXIFTOOL] [-csv EXTRACTCSV] [-d]

Extract and visualize Flir Image data

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input image. Ex. img.jpg
  -p, --plot            Generate a plot using matplotlib
  -c {flir,flir_high_contrast,fluke,fluke_high_contrast,hot,inferno,oxy,plasma,seismic,thermal}, --colormap {flir,flir_high_contrast,fluke,fluke_high_contrast,hot,inferno,oxy,plasma,seismic,thermal}
                        Colormap to use for plotting. Default: thermal
  -exif EXIFTOOL, --exiftool EXIFTOOL
                        Custom path to exiftool
  -csv EXTRACTCSV, --extractcsv EXTRACTCSV
                        Export the thermal data per pixel encoded as csv file
  -d, --debug           Set the debug flag
```

This command will show an interactive plot of the thermal image using matplotlib and create two image files such as *flir_example_thermal.png* and *flir_example_rgb_image.jpg*.
Both are RGB images, while the original temperature array is available using the `get_thermal_np` or `export_thermal_to_csv` functions.

The selected colormap is used both for the interactive plot and for the saved thermal PNG.

The functions `get_rgb_np` and `get_thermal_np` yield numpy arrays and can be called from your own script after importing this lib.

## Colormaps

Available colormaps:

- `thermal`
- `oxy`
- `inferno`
- `flir`
- `flir_high_contrast`
- `fluke`
- `fluke_high_contrast`
- `seismic`
- `hot`
- `plasma`

Colormap definitions are stored in `colormaps.py`. The custom `.npy` colormaps are loaded there so the main extractor stays focused on extraction and plotting logic.

## Interactive Plot Controls

The interactive plot now includes buttons directly in the Matplotlib window:

- `Add Marker`: place a normal temperature marker
- `Set Reference`: place the delta reference marker
- `Add Delta`: place a marker that shows delta relative to the reference marker
- `Delete Marker`: click an existing marker to remove it

Behavior:

- only one reference marker is active at a time
- if the reference marker is deleted, all delta markers show `Δ?`
- if a new reference marker is added, all delta markers are recomputed automatically

The plot also includes sliders for:

- `Scale Min`
- `Scale Max`
- `Sharpen`

These allow quick visual tuning without editing the code.

## Supported/Tested cameras:

- Flir One (thermal + RGB)
- Xenmuse XTR (thermal + thumbnail, set the subject distance to 1 meter)
- AX8 (thermal + RGB)
- Flir E75

Other cameras might need some small tweaks (the embedded raw data can be in multiple image formats)

## Credits

Raw value to temperature conversion is ported from this R package: https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
Original Python code from: https://github.com/Nervengift/read_thermal.py

