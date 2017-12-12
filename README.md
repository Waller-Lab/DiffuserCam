# DiffuserCam Algorithms
This code is based on the paper "DiffuserCam: Lensless Single-exposure 3D Imaging" available on [arXiv](https://arxiv.org/abs/1710.02134). It implements the alternating direction method of multipliers (ADMM) algorithm described in the paper for recovering 3D volumes from 2D raw data captured with DiffuserCam.


### Running the solver
```
xhat = DiffuserCam_main('DiffuserCam_settings.m');
```
The settings file contains all user-controlled parameters.

### Data Files
Two files are necessary to use the solver.
1. Stack of point spread functions (PSFs) taken at different axial distances, saved as '.mat' file with any background subtracted
2. Raw data, saved as image (.png)

Paths to these files are specified in the settings file.

### Included Example
To allow users to experiment with the code without building their own DiffuserCam system, we've provided an example PSF stack and raw data. The raw data is for a USAF resolution target, placed at an angle in front of the sensor. In this example, the target is imaged from the back, so you should see a mirror image of a USAF target after solving. Note that we've downsampled these images to reduce the file sizes and speed up the code, so the quality of the results is lower compared to higher resolutions.
