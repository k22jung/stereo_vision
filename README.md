# Stereo Vision

**Stereo Vision** is a program for stereo vision and generating low-noise disparity maps to detect nearby objects, such as pedistrians and vehicles. The algorithms used are SGBM (Semi Global Block Matching) and the WLS (Weighted Least Squares) Filter. It works for a single pair of frames captured by stereo cameras.



<p align="center"> 
<img src="https://github.com/k22jung/stereo_vision/blob/master/output/disparity_filtered.jpg">
</p>
<p align="center"> 
<img src="https://github.com/k22jung/stereo_vision/blob/master/output/left_image.jpg">
</p>

The original code that this project is based off of can be found [here](https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/disparity_filtering.cpp). 

The original colored images are from the [KITTI Object Tracking Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset.
 

## Dependencies

The program was ran and created for Ubuntu 16.04
- [OpenCV 3.2.0](http://opencv.org/releases.html)

You must build OpenCV with extra modules, following these [instructions](https://github.com/opencv/opencv_contrib). The latest release for the extra modules can be found [here](https://github.com/opencv/opencv_contrib/releases).

## Running

This is an Eclipse project, you may simply compile it or use `make` in the `Debug` folder. Run with argument `-h` for details on how to use.




