# CorrSearch Software

### Description (May 26 2015)
This software implements a target-search for first-person point-of-view videos. Visit [our project page](http://yonetaniryo.github.io/corrsearch/) to see the original paper presented at CVPR.


### References
Ryo Yonetani, Kris M. Kitani and Yoichi Sato: "Ego-Surfing First-Person Videos", in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

### Dependencies
- Python (2.7.6-)
   - IPython (3.1.0-), Numpy (1.9.2-), Scipy (0.15.1-), Scikit-learn (0.16.1-), Scikit-image (0.11.3-) and OpenCV (2.4.9-)
- [LIBSVX](http://www.cse.buffalo.edu/~jcorso/r/supervoxels/)

### How to run this software
- See an example on [IPython notebook page](https://github.com/yonetaniryo/corrsearch/blob/master/ipython_notebook/example.ipynb) or execute  ```ipython notebook``` on ```ipython_notebook``` directory.
- This project currently includes the only target localization function. Please ask us if you need the program to train a generic targetness (e.g., when training on videos at different resolutions is needed).
