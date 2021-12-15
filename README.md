# Transformio

Transformio creates, applies, and evaluates coordinate transformations for vector and raster geospatial data. 


## Introduction

The geospatial Python stack currently lacks a complete library for vector and raster coordinate transformation. 

For vector geometries, the `pyproj` package supports transforming coordinates from one geographic or projected coordinate system (e.g. WGS84) to another (e.g. World Robinson). This is sufficient for many, but not all, cases of vector geometry transformation. 

For raster grids, the `gdal` or `rasterio` packages can be used to register or rectify scanned map images or remote sensing imagery based on a set of ground control points. These packages, all based on the GDAL library, allow basic transformation based on 1st to 3rd order polynomials or thin-plate-splines. 

These use-cases, while common, only cover a small specter of what's possible when it comes to coordinate transformation. Additionally, relying on heavy libraries like GDAL precludes their usage in highly portable environments, especially when the same can be accomplished using only `numpy` and/or `pyproj`. There is also a lack of tools or utilities to analyze the inevitable accuracy or error resulting from the transformation process. 

The purpose of this library is to provide a lightweight, fast, customizable, fully-featured, and cutting-edge transformation library, with convenience functions for common user-operations and accuracy analysis. All it requires is `numpy`, which makes it lighting fast, and `pyproj` if geographic transformations are required. 


## How it works

Start by importing transformio:

    >>> import transformio as tio

Transformio works by providing a set of classes for various types of transformations that can be used to predict or transform a series of input coordinates. For instance: 

    >>> # create an affine transform that offsets the coordinates by 10
    >>> trans = tio.transforms.Affine(offset=(10,10))
    >>> # transform a set of input coordinates
    >>> x = [1,1,5]
    >>> y = [1,5,5]
    >>> xpred,ypred = trans.predict(x, y)
    >>> xpred
    array([11., 11., 15.])
    >>> ypred
    array([11., 15., 15.])

Many transformations are created by fitting them to a set of observed input points and their equivalent points in the output coordinate system. So if we fit a new transform based on the input and output coordinates of the transform we created previously, we should end up with approximately the same transform: 

    >>> transfit = tio.transforms.Affine()
    >>> transfit.fit(x, y, xpred, ypred)
    Affine Transform(estimated=True)
    >>> trans.A
    array([[ 1.,  0., 10.],
        [ 0.,  1., 10.],
        [ 0.,  0.,  1.]])
    >>> transfit.A.round(14)
    array([[ 1.,  0., 10.],
        [ 0.,  1., 10.],
        [ 0.,  0.,  1.]])

All transformations have a JSON-compatible dictionary representation:

    >>> import json
    >>> datastring = json.dumps(trans.to_json())
    >>> datastring
    '{"type": "Affine", "params": {}, "data": {"A": [[1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]]}}'

This allows transformations to be easily stored between sessions or transferred between application:

    >>> recreated = tio.transforms.from_json(json.loads(datastring))
    >>> trans.A
    array([[ 1.,  0., 10.],
        [ 0.,  1., 10.],
        [ 0.,  0.,  1.]])
    >>> recreated.A
    array([[ 1.,  0., 10.],
        [ 0.,  1., 10.],
        [ 0.,  0.,  1.]])


## Transforming vector geometries

### *Coordinate reprojection*

So for the typical task of reprojecting a vector geometry, you would simply use the `transforms.MapProjection` transformation. While this essentially just uses the `pyproj` library in the background, we provide convenience functions for applying this to entire GeoJSON geometries:

    >>> # define a geojson of points for every 10 lat/long coordinates
    >>> points = []
    >>> for y in range(-90, 90, 10):
    ...     for x in range(-180, 180+1, 10):
    ...         points.append((x,y))
    >>> geoj = {'type':'MultiPoint', 'coordinates':points}

    >>> # define a transformation from lat/long to an orthographic projection
    >>> # ...of the globe as seen from space
    >>> fromcrs = '+proj=longlat +datum=WGS84 +no_defs'
    >>> tocrs = '+proj=ortho +lat_0=-10 +lon_0=30 +x_0=30 +y_0=-10'
    >>> trans = tio.transforms.MapProjection(fromcrs, tocrs)

    >>> # transform the points geojson
    >>> newgeoj = tio.vector.transform(geoj, trans)

Let's visualize this on top of a background layer to see what it looks like:

    >>> # load the background layer
    >>> from PIL import Image
    >>> background = Image.open('tests/data/land_shallow_topo_2048.png')
    >>> bounds = [-180,90,180,-90] # left,upper,right,bottom

    >>> # render the original on top of the background
    >>> im = background.copy()
    >>> im = tio.utils.draw_geojson(geoj, im, bounds, fillcolor="red")
    >>> im.save('tests/output/doctest-vector-gridpoints.png')

    >>> # transform the background layer
    >>> im = background.copy()
    >>> im2geo = tio.imwarp.fitbounds(*im.size, bounds)
    >>> imtrans = tio.transforms.Chain([im2geo,trans])
    >>> im,affine = tio.imwarp.warp(im, imtrans)
    >>> bounds = tio.imwarp.imbounds(*im.size, imtrans)

    >>> # render the transform on top of the background
    >>> im = tio.utils.draw_geojson(newgeoj, im, bounds, fillcolor="red")
    >>> im.save('tests/output/doctest-vector-reprojection.png')

Original                   |  Transformed
:-------------------------:|:-------------------------:
![Original image](/tests/output/doctest-vector-gridpoints.png) | ![Expected image](/tests/output/doctest-vector-reprojection.png)

### *Simple geometry adjustments*

In other cases, one might want to apply non-geographic/projection-based transformations, which is beyond the scope of the `pyproj` library. For instance, one might want to simply rotate, scale, or skew a set of geometries, e.g. for artistic or visualization purposes. This can be done easily using the `Affine` transform: 

    >>> # transform the background layer
    >>> im = background.copy()
    >>> trans = tio.transforms.Affine(rotate=45)
    >>> imtrans = tio.transforms.Chain([im2geo,trans])
    >>> im,affine = tio.imwarp.warp(im, imtrans)
    >>> bounds = tio.imwarp.imbounds(*im.size, imtrans)

    >>> # transform the points geojson
    >>> newgeoj = tio.vector.transform(geoj, trans)

    >>> # render the transform on top of the background
    >>> im = tio.utils.draw_geojson(newgeoj, im, bounds, fillcolor="red")
    >>> im.save('tests/output/doctest-vector-rotate.png')

Original                   |  Transformed
:-------------------------:|:-------------------------:
![Original image](/tests/output/doctest-vector-gridpoints.png) | ![Expected image](/tests/output/doctest-vector-rotate.png)

### *Digitized geometry transformation*

A more practical example is when digitizing the data contents of a scanned map image or remote sensing imagery. In these cases the extracted geometry coordinates are sometimes stored as pixel coordinates, but needs to be transformed to geographic space based on the georeferencing information of the source imagery. For instance, if the source image coordinate system is defined by a 2nd order polynomial function, the extracted vector geomtries are transformed as follows:

... 

### *Dataset integration and bias correction*

One of the cool things this allows, is easily integrating one dataset with another dataset, e.g. if through visual inspection we see that one of the datasets is biased or offset in some way. One of the ways this bias can be determined is by measuring the closest point in another reference dataset that we know to be correct:

...

We can then estimate a transform based on these equivalent points: 

...

Finally, the resulting transform can be used to correct the biased dataset: 

... 


## Transforming raster grids

For raster datasets which consist of regularly spaced grid cells, each cell or image pixel needs to be projected and resampled to another coordinate system. Such raster transformation is more complicated and involves more steps than in the case of vector transformation. Transformio greatly simplifies this process, and works seamlessly with the Pillow imaging library. 

There are two basic steps to this process that transformio handles behind the scenes. 

1. The first is the process of mapping coordinates between the two coordinate systems. Forward or backward... 

2. The second is how to resample and interpolate the pixels. Currently, transformio only supports nearest neighbour resampling... 

### *Raster reprojection*

One common use-case is just reprojecting a raster dataset from one projection to another. 

First, we need to define the affine transform needed to determine the geographic coordinates at each pixel. Typically, this is stored as six parameters in the metadata of the raster file. For instance: 

    >>> # load a raster dataset
    >>> from PIL import Image
    >>> im = Image.open('tests/data/land_shallow_topo_2048.png')
    
    >>> # load affine parameters from .wld file
    >>> with open('tests/data/land_shallow_topo_2048.wld', mode='r') as f:
    ...     a,b,c,d,e,f = f.read().split()
    ...     a,b,c,d,e,f = map(float, [a,b,c,d,e,f])

    >>> # create image-to-geographic transform from affine parameters
    >>> A = [[a,b,c],[d,e,f],[0,0,1]]
    >>> img2geo = tio.transforms.Affine(A=A)
    >>> img2geo.A.round(10).tolist()
    [[0.17578125, 0.0, -180.0], [0.0, -0.17578125, 90.0], [0.0, 0.0, 1.0]]

![Original image](/tests/data/land_shallow_topo_2048.png)

Alternatively, if the affine parameters are not explicitly provided, you may instead generate the transform from the known coordinate bounds of the dataset: 

    >>> # create image-to-geographic transform from coordinate bounds
    >>> bounds = [-180,90,180,-90] # left,upper,right,bottom
    >>> img2geo = tio.imwarp.fitbounds(im.size[0], im.size[1], bounds)
    >>> img2geo.A.round(10).tolist()
    [[0.17578125, 0.0, -180.0], [0.0, -0.17578125, 90.0], [0.0, 0.0, 1.0]]

Next, we define a map projection transform for projecting the coordinate reference system of the original raster dataset into the target reference system: 

    >>> # define original to target projection transform
    >>> fromcrs = '+proj=longlat +datum=WGS84 +no_defs' #'epsg:4326'
    >>> tocrs = 'esri:54009'
    >>> geo2proj = tio.transforms.MapProjection(fromcrs, tocrs)

That's all we need. Now just call on the `warp` function along with a list of specifying that these two transforms should be called in sequence - first from image to source geographic space then from source to target geographic space: 

    >>> # warp the image
    >>> warped,affine = tio.imwarp.warp(im, [img2geo,geo2proj])
    >>> warped.save('tests/output/doctest-raster-reprojection.png')

![Expected image](/tests/output/doctest-raster-reprojection.png)

This returns a numpy image array containing the warped image data, and the affine transform parameters defining the image's coordinate system. 


### *Map georeferencing/registration*

Another common scenario is when performing georeferencing of scanned map images. Let's say we have been given a set of ground control points defining the real-world coordinates at different locations in a scanned map image:

    >>> # load the image and control points
    >>> im = Image.open('tests/data/burkina_pol96.jpg')
    >>> impoints = [(574, 304), (285, 854), (816, 934), (945, 96), (522, 114), (779, 241), (841, 302), (918, 384), (102, 411), (316, 444)]
    >>> geopoints = [(-0.86537, 14.22963), (-3.279831, 9.6586821), (1.133333, 8.983333), (2.4022, 15.9182), (-1.3094536, 15.8179117), (0.917385, 14.730746), (1.454179, 14.207113), (2.1098, 13.51366), (-4.895615, 13.303346), (-3.0694, 13.0725)]

    >>> # visualize on a map
    >>> geoj = {'type':'MultiPoint', 'coordinates':impoints}
    >>> w,h = im.size
    >>> imbounds = [0,0,w,h]
    >>> draw = im.copy()
    >>> tio.utils.draw_geojson(geoj, draw, imbounds, fillcolor="red")
    >>> draw.save('tests/output/doctest-map-controlpoints.png')

![Original image](/tests/output/doctest-map-controlpoints.png)

Based on these points we want to fit the map using a 2nd order polynomial transformation: 

    >>> # create and fit the transform model
    >>> imx,imy = zip(*impoints)
    >>> geox,geoy = zip(*geopoints)
    >>> trans = tio.transforms.Polynomial(order=2)
    >>> trans.fit(imx, imy, geox, geoy)
    Polynomial Transform(order=2, estimated=True)

Now just supply this transform to the `warp` function:

    >>> # warp the image
    >>> warped,affine = tio.imwarp.warp(im, trans)

Let's see how the transformed map compares with the real-world geo-coordinates:

    >>> geoj = {'type':'MultiPoint', 'coordinates':geopoints}
    >>> bounds = tio.imwarp.imbounds(*warped.size, trans)
    >>> tio.utils.draw_geojson(geoj, warped, bounds, fillcolor="red")
    >>> warped.save('tests/output/doctest-map-georeferencing.png')

![Expected image](/tests/output/doctest-map-georeferencing.png)


### *Remote sensing image registration/rectification*

Remote sensing imagery from satellites or aircraft similarly needs to undergo a process of registration or rectification that adjusts and defines the coordinates of the image. Due to the large volumes of imagery this is often done automatically using sophisticated algorithms to identify control points based on pixel correlations across imagery. The output control points of these automated algorithms are then used to fit a transformation function and warp the image in the same way as before:

... 

More advanced cases use information about elevation and camera perspective to correct various distortions, but this is not yet supported here. 


## Supported transformations

The goal of transformio is to support a wide range of transformations useful to the geospatial field, including the latest cutting-edge algorithms emerging from the literature. Below are some usage-examples for the currently supported transformations. 

First, let's define the map image we will be using: 

    >>> # load the image and control points
    >>> im = Image.open('tests/data/china_pol96.jpg')
    >>> impoints = [(532, 64), (113, 112), (230, 161), (289, 107), (1018, 166), (611, 253), (108, 379), (866, 416), (1006, 470), (404, 502), (933, 583), (75, 645), (332, 638), (413, 694), (869, 701), (913, 717), (852, 738), (548, 749), (760, 779), (408, 924), (651, 935), (949, 942)]
    >>> geopoints = [(101.621839, 56.161959), (71.44598, 51.1801), (80.26669, 50.42675), (83.76361, 53.36056), (135.08379, 48.48272), (106.88324, 47.90771), (68.04073, 33.12699), (119.70478, 31.94689), (129.04028, 35.10278), (94.900606, 36.406717), (121.05804, 29.32955), (78.715422, 21.426482), (91.1000101308, 29.6450238231), (96.86525, 21.09148), (121.7423789, 24.7184669), (121.56833333333, 25.03583333333), (118.080017048, 24.4499920847), (98.70707, 23.43771), (113.325010131, 23.1449813019), (111.2626075, 1.0875755), (107.59546, 16.4619), (125.567222, 8.805556)]

    >>> # create and fit the transform model
    >>> imx,imy = zip(*impoints)
    >>> geox,geoy = zip(*geopoints)

### *Map Projection transformations*

Map projection transforms are specified using the proj4 strings of the source and target projection: 

    >>> # define an approximate pixel to original projection transform
    >>> img2geo = tio.transforms.Affine()
    >>> img2geo.fit(imx, imy, geox, geoy)
    
    >>> # define original to target projection transform
    >>> fromcrs = '+proj=longlat +datum=WGS84 +no_defs ' #'epsg:4326'
    >>> tocrs = '+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs'
    >>> geo2proj = tio.transforms.MapProjection(fromcrs, tocrs)

    >>> # warp the image
    >>> warped,affine = tio.imwarp.warp(im, [img2geo,geo2proj])
    >>> warped.save('tests/output/doctest-transforms-mapprojection.png')

![Expected image](/tests/output/doctest-transforms-mapprojection.png)

### *Similarity transformations*

...

### *Affine transformations*

...

### *Projective transformations*

...

### *Polynomial transformations*

Polynomial transforms are typically performed using the 1st, 2nd, or 3rd order, with higher orders providing a tighter fit to the data:

    >>> for order in [1,2,3]:
    ...     trans = tio.transforms.Polynomial(order=order)
    ...     trans.fit(imx, imy, geox, geoy)
    ...     # warp the image
    ...     warped,affine = tio.imwarp.warp(im, trans)
    ...     warped.save('tests/output/doctest-transforms-polynomial-{}.png'.format(order))

![Expected image](/tests/output/doctest-transforms-polynomial-1.png)

![Expected image](/tests/output/doctest-transforms-polynomial-2.png)

![Expected image](/tests/output/doctest-transforms-polynomial-3.png)

If you don't specify the order of your polynomial transform, this will be determined automatically based on the minimum number of required control points. 

### *Triangulated Irregular Network transformations*

TIN (triangulated irregular network) transforms, also known as the piecewise affine transform, divides the image into a mesh of triangles formed by the control points and inside each triangle estimates a local affine transforms for transforming the pixels contained within. When using this transform it's important to include control points along the edges and corners, since the transform is only defined inside the bounding box of the input control points and will result in a cropped image. 

    >>> # create and fit the transform model
    >>> trans = tio.transforms.TIN()
    >>> trans.fit(imx, imy, geox, geoy)
    TIN Transform(estimated=True)

    >>> # warp the image
    >>> warped,affine = tio.imwarp.warp(im, trans)
    >>> warped.save('tests/output/doctest-transforms-tin.png')

![Expected image](/tests/output/doctest-transforms-tin.png)

### *Chained transformations*

...


## Accuracy evaluation

An important but often overlooked aspect of coordinate transformation such as map georeferencing is the impact it has on accuracy or error. Transformio has key functionality for evaluating the accuracy of a transformation, so that the overall confidence in the final product can be evaluated. This functionality is contained in the `accuracy` module. 

...


## Control point outliers

The ground control points used to fit transformation models are often noisy and contain outliers, resulting in an adverse effect on the fitted transform. Therefore, we need efficient and automated ways to detect and exclude these. Transformio includes a set of functions to help with this. 

...


## Model selection

In addition to evaluating individual models, transformio provides several algorithms for automatically comparing and selecting the most optimal transformation model for a particular use-case. 

...
