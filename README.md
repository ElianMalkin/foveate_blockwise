# foveate_blockwise
## Real-time image and video foveation transform using PyCUDA 

Foveation implementation using adaptive Gaussian blurring optimized for real-time performance. 
The algorithm exploits the CUDA architecture to generate the foveated image in blocks of varying blurring strength. 

See it in action: https://youtu.be/Rr5oaiIsVbA

## What is Block-wise Foveation? 

![blockwise_approach](docs/images/blockwise_approach.png)

Our visual field is non-uniform. Acuity is high at the fovea, where cone photoreceptors are closely packed, allowing us to discern fine details. Cone density drops off sharply towards the periphery, giving us lower spatial acuity but greater awareness of our surroundings with a wide field of view. 

Block-wise foveation enables real-time experimentation with spatial frequency models of human or animal retinas. It is built around the CUDA architecture, utilizing the parallel processing power of the GPU to perform spatially-variant Gaussian blur on the image frame. 

Eye trackers enable us to discern the fixation point of the user and allocate resources appropriately, be it the rendering workload in virtual reality applications, or compression strength for streaming video. The flexibility and real-time performance of block-wise foveation supports psychophysical experiments to determine the parameters of a foveation pipeline, and serves as a showcase of the suitability of the CUDA architecture for GPU-accelerated foveation. 

## Getting Started

Blurring strength throughout the image frame can be defined in one of two ways:

1. A circularly-symmetric function can be used to define the spatial frequency falloff with eccentricity from the fixation point - an implementation is provided based on parameters and psychometric functions sourced from [Wilson S. Geisler, Jeffrey S. Perry, "Real-time foveated multiresolution system for low-bandwidth video communication," Proc. SPIE 3299, Human Vision and Electronic Imaging III, (17 July 1998)](http://www.svi.cps.utexas.edu/spie1998.pdf).

2. A greyscale image can be used as a map of retinal ganglion cell (RGC) density distribution and therefore the blurring strength across the image frame. 

*Example of greyscale RGC maps and their foveation transforms:*
![map example](docs/images/rgc_mosaic_env.png)

The fixation point (center of gaze) can be displaced anywhere in the visual field. 

We provide three files:

1. **foveate_blockwise.py:** Foveates and displays/saves a single image from the `/images` directory. 
2. **foveate_blockwise_track.py:** A real-time foveation demo where the fixation point follows the mouse cursor. 
3. **foveate_blockwise_draw.py:** Similar to the tracking demo, but the user first draws a greyscale RGC mapping before seeing it in action on an image. 
<!--More information, including a detailed algorithm description and suggestions for modifications, is available here.-->

## Install

This implementation requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [PyCUDA wrapper](https://pypi.org/project/pycuda/).

PyCUDA and other requirements can be installed using pip:

```
pip install -r requirements.txt
```

## Run

To foveate a single image:
```
python src/foveate_blockwise.py -v
```
To foveate and save a particular image, place it in the `/images` directory, then specify its name with the `-i` parameter. To save the image use the `-o` option, and provide the output directory and filename:
```
python src/foveate_blockwise.py -i my_image.jpg -o output/fov_image.png
```

To run the tracking demo:
```
python src/foveate_blockwise_track.py 
```
To run the drawing demo:
```
python src/foveate_blockwise_draw.py
```

Other available options for each file can be found with the `-h` parameter:

* `-h, --help`:          Displays help
* `-p, --gazePosition`:  Gaze position coordinates, (vertical down) then (horizontal right), (e.g. `-p 512,512`)
* `-f, --fragmentSize`:  Width and height of fragments for foveation, (e.g. `-f 16,16`), default: 32 x 32
* `-v, --visualize`:     Show foveated images
* `-i, --inputFile`:     Input image from "images" folder
* `-o, --outputDir`:     Output directory and filename

## Citation

If you found this code useful, consider citing:
```
@misc{BlockwiseFoveation,
author = {Malkin, Elian and Deza, Arturo and Poggio, Tomaso},
title = {{CUDA}-{Optimized} real-time rendering of a {Foveated} {Visual} {System}}
year = {2020},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/ElianMalkin/foveate_blockwise}}
}
```

## License 

MIT License.

