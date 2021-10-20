# Transformio

Creates, applies, and evaluates coordinate transformations. 


## Introduction

The geospatial Python stack currently lacks a complete library for vector and raster coordinate transformation. 

For vector geometries, the `pyproj` package supports transforming coordinates from one geographic or projected coordinate system (e.g. WGS84) to another (e.g. World Robinson). This is sufficient for most, but not all, cases of vector geometry transformation. 

For raster grids, the `gdal` or `rasterio` packages can be used to register or rectify scanned map images or remote sensing imagery based on a set of ground control points. These packages, all based on the GDAL library, allow basic transformation based on 1st to 3rd order polynomials or thin-plate-splines. 

These use-cases, while common, only cover a small specter of what's possible when it comes to coordinate transformation. Additionally, relying on heavy libraries like GDAL precludes their usage in highly portable environments, especially when the same can be accomplished using only `numpy` and/or `pyproj`. There is also a lack of tools or utilities to analyze the inevitable accuracy or error resulting from the transformation process. 

The purpose of this library is to provide a lightweight, fast, customizable, fully-featured, and cutting-edge transformation library, with convenience functions for common user-operations and accuracy analysis. All it requires is `numpy`, which makes it lighting fast, and `pyproj` if geographic transformations are required. 


## Quick start

Start by importing transformio:

>>> import transformio as tio

Transformio works by providing a set of classes for various types of transformations that can be used to predict or transform a series of input coordinates. For instance: 

>>> # create an affine transform that offsets the coordinates by 10
>>> xoff = 10
>>> yoff = 10
>>> A = [[1,0,xoff],
...      [0,1,yoff],
...      [0,0,1]]
>>> trans = tio.transforms.Polynomial(A=A)
>>> # transform a set of input coordinates
>>> x = [1,1,1,1]
>>> y = [2,2,2,2]
>>> xpred,ypred = trans.predict(x, y)
>>> xpred
[11,11,11,11]
>>> ypred
[12,12,12,12]

Most transformations are created by fitting them to a set of observed input points and their equivalent points in the output coordinate system: 

... 

All transformations have a JSON-compatible dictionary representation that allows them to be easily stored between sessions or transferred between application:

... 


### Transforming vector geometries

#### Reprojection

So for the typical task of reprojecting a vector geometry, you would simply use the `transforms.Projection` transformation. While this essentially just uses the `pyproj` library in the background, we provide convenience functions for applying this to entire GeoJSON geometries:

...

In other cases, one might want to apply non-geographic/projection-based transformations to vector geometries, which is not currently possible with `pyproj`. 

#### Simple vector adjustments

For instance, one might want to simply rotate, scale, or skew a set of geometries, e.g. for artistic or visualization purposes. This can be done easily using the 1st order `Polynomial` transform, also known as affine transform: 

... 

#### Digitized geometry transformation

A more practical example is when digitizing the data contents of a scanned map image or remote sensing imagery. In these cases the extracted geometry coordinates are sometimes stored as pixel coordinates, but needs to be transformed to geographic space based on the georeferencing information of the source imagery. For instance, if the source image coordinate system is defined by a 2nd order polynomial function, the extracted vector geomtries are transformed as follows:

... 

#### Dataset integration bias correction

One of the cool things this allows, is easily integrating one dataset to another dataset, e.g. if through visual inspection we see that one of the datasets is biased or offset in some way. One of the ways this bias can be determined is by measuring the closest point in another reference dataset that we know to be correct, and estimating a transform based on these equivalent points. The resulting transform can then be used to correct the biased dataset: 

... 

#### Vector shape morphing

This same general idea can be used to gradually morph a vector geometry towards another shape. After calculating the transform, a weighting factor can be used to move the vertices only part of the way: 

... 


### Transforming raster grids

For raster datasets which consist of regularly spaced grid cells, each cell or image pixel needs to be projected and resampled to another coordinate system. Such raster transformation is more complicated and involves more steps than in the case of vector transformation. Transformio greatly simplifies this process, and works seamlessly with the Pillow imaging library. 

#### Raster reprojection

One common use-case is just reprojecting a raster dataset from one projection to another: 

>>> # load a satellite image
>>> from PIL import Image
>>> im = Image.open('tests/data/land_shallow_topo_2048.tif')
>>> # define corner coordinates in pixel and geographic space
>>> imgcorners = [(0,0),(im.size[0],0),im.size,(0,im.size[1])]
>>> geocorners = [(-180,90),(180,90),(180,-90),(-180,-90)]
>>> x,y = zip(*imgcorners)
>>> x2,y2 = zip(*geocorners)
>>> # based on these create the image to geographic transform
>>> img2geo = tio.transforms.Polynomial(order=1)
>>> img2geo.fit(x, y, x2, y2)
>>> geo2img = tio.transforms.Polynomial(order=1)
>>> geo2img.fit(x, y, x2, y2, invert=True)
>>> # define the geographic to projection transform
>>> fromcrs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
>>> tocrs = '+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs'
>>> geo2proj = tio.transforms.Projection(fromcrs, tocrs)
>>> proj2geo = tio.transforms.Projection(tocrs, fromcrs)
>>> # create the final chained transforms
>>> forw = tio.transforms.Chain([img2geo,geo2proj])
>>> back = tio.transforms.Chain([proj2geo,geo2img])
>>> # warp the image
>>> warped,affine = tio.imwarp.warp(im, forw, back)
>>> warped.save('tests/output/raster-reprojection.png')

This returns a numpy image array containing the warped image data, and the affine transform parameters defining the image's coordinate system. 

#### Map georeferencing/registration

Another common scenario is when performing georeferencing of scanned map images. Let's say we have been given a set of ground control points defining the real-world coordinates at different locations in a scanned map image, and want to fit the map to these points based on a 2nd order polynomial transformation: 

>>> # load the image and control points
>>> im = Image.open('scanned-map.png')
>>> gcps = [()]
>>> # create and fit the transform model
>>> trans = tio.transforms.Polynomial(order=2)
>>> x,y = zip(*gcps)
>>> trans.fit(x, y)
>>> # warp the image
>>> warped,affine = tio.imwarp.warp(im, trans, trans.inverse())


#### Remote sensing image registration/rectification

Remote sensing imagery from satellites or aircraft similarly needs to undergo a process of registration or rectification that adjusts and defines the coordinates of the image. Due to the large volumes of imagery this is often done automatically using sophisticated algorithms based on pixel correlations with imagery of known location. The workflow uses the same exact approach as before:

... 


## Supported transformations

The goal of transformio is to support all available transformations, including the latest cutting-edge algorithms emerging from the literature. The following transformations are currently supported: 

### Projection transformations

### Polynomial transformations

### Triangulated Irregular Network transformation

### Chained transformations


## Accuracy evaluation

An important but often overlooked aspect of coordinate transformation such as map georeferencing is the impact it has on accuracy or error. Transformio has key functionality for evaluating the accuracy of a transformation, so that the overall confidence in the final product can be evaluated. This functionality is contained in the `accuracy` module. 


## Model selection

In addition to evaluating individual models, transformio provides several algorithms for automatically selecting the most optimal transformation model for a particular use-case. 
