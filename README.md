# Representative Isovalue Detection in Volumetric Data

Implementing a new measure on volumetric data to detect relevant iso-values using the published paper ”Representative Isovalue
Detection and Isosurface Segmentation Using Novel Isosurface Measures” DOI:10.1111/cgf.13961 [link to paper](https://www.researchgate.net/publication/343051674_Representative_Isovalue_Detection_and_Isosurface_Segmentation_Using_Novel_Isosurface_Measures)
Replicated results in the paper and used them further for Interactive Computer Graphics project of rendering the found iso-value surfaces using Ray Tracing on Metal.

The repository also includes a Python script to calculate the gradient of the image read from folder.


Install
=======

Install python3 

Using following libraries:
- numpy 
- matplotlib
- skimage 
- scipy


Usage
=====

Run script <kbd>python3 imageGradientCalculation.py </kbd> to calculate gradient of image. 

Run script <kbd>python3 mainIsosurfaces.py </kbd> to find relevant iso-values of the given data. 
