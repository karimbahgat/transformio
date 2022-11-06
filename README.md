# Transformio

Transformio creates, applies, and evaluates coordinate transformations for vector and raster geospatial data. 

WARNING: This library is still in early development, so the final API has not yet been settled on and currently contains several unresolved bugs. 


## Table of Contents

- [Introduction](#introduction)
- [How it works](#how-it-works)
- [Supported transformations](#supported-transformations)
- [Accuracy evaluation](#accuracy-evaluation)
- [Control point outliers](#control-point-outliers)
- [Model selection](#model-selection)
- [Vector examples](#vector-examples)
- [Raster examples](#raster-examples)


## Introduction

The geospatial Python stack currently lacks a complete library for vector and raster coordinate transformation. 

For vector geometries, the `pyproj` package supports transforming coordinates from one geographic or projected coordinate system (e.g. WGS84) to another (e.g. World Robinson). This is sufficient for many, but not all, cases of vector geometry transformation. 

For raster grids, the `gdal` or `rasterio` packages can be used to register or rectify scanned map images or remote sensing imagery based on a set of ground control points. These packages, all based on the GDAL library, allow basic transformation based on 1st to 3rd order polynomials or thin-plate-splines. 

These use-cases, while common, only cover a small specter of what's possible when it comes to coordinate transformation. Additionally, relying on heavy libraries like GDAL precludes their usage in highly portable environments, especially when the same can be accomplished using only `numpy` and/or `pyproj`. There is also a lack of tools or utilities to analyze the inevitable accuracy or error resulting from the transformation process. 

The purpose of this library is to provide a lightweight, fast, customizable, fully-featured, and cutting-edge transformation library, with convenience functions for common user-operations and accuracy analysis. All it requires is `numpy`, which makes it lighting fast, and `pyproj` if geographic transformations are required. 


## How it works

Start by importing transformio:

    >>> import transformio as tio

### Transforming arrays of xy coordinates

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

### Inverse transformations

Transforms can also be inversed, which means that it's possible to recreate the original coordinates from the predicted ones:

    >>> inv = trans.inverse()
    >>> xinv,yinv = trans.predict(xpred, ypred)

    >>> xinv
    array([1., 1., 5.])

    >>> yinv
    array([1., 5., 5.])

### Fitting transformations based on observed data

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

### Serializing and deserializing transformations

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


## Supported transformations

The goal of transformio is to support a wide range of transformations useful to the geospatial field, including the latest cutting-edge algorithms emerging from the literature. Below are some usage-examples for the currently supported transformations. 

First, let's define the map image we will be using: 

    >>> # load the image and control points
    >>> im = Image.open('tests/data/china_pol96.jpg')

    >>> impoints = [(532, 64), (113, 112), (230, 161), (289, 107), (1018, 166), (611, 253), (108, 379), (866, 416), (1006, 470), (404, 502), (933, 583), (75, 645), (332, 638), (413, 694), (869, 701), (913, 717), (852, 738), (548, 749), (760, 779), (408, 924), (651, 935), (949, 942)]

    >>> geopoints = [(101.621839, 56.161959), (71.44598, 51.1801), (80.26669, 50.42675), (83.76361, 53.36056), (135.08379, 48.48272), (106.88324, 47.90771), (68.04073, 33.12699), (119.70478, 31.94689), (129.04028, 35.10278), (94.900606, 36.406717), (121.05804, 29.32955), (78.715422, 21.426482), (91.1000101308, 29.6450238231), (96.86525, 21.09148), (121.7423789, 24.7184669), (121.56833333333, 25.03583333333), (118.080017048, 24.4499920847), (98.70707, 23.43771), (113.325010131, 23.1449813019), (111.2626075, 1.0875755), (107.59546, 16.4619), (125.567222, 8.805556)]

    >>> # split input and output coordinates into x and y arrays
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

### *Affine transformations*

Affine transformations allow for simple transformations such as offset, scale, skew, and rotation: 

    # offset the x axis by 10
    >>> trans = tio.transforms.Affine(offset=(10,0))
    >>> warped,affine = tio.imwarp.warp(im, trans)
    >>> warped.save('tests/output/doctest-transforms-affine-offset.png')

    # scale the x axis to double length
    >>> trans = tio.transforms.Affine(scale=(2,1))
    >>> warped,affine = tio.imwarp.warp(im, trans)
    >>> warped.save('tests/output/doctest-transforms-affine-scale.png')

    # skew
    # ... 

    # rotate image by 45 degrees
    >>> trans = tio.transforms.Affine(rotate=45)
    >>> warped,affine = tio.imwarp.warp(im, trans)
    >>> warped.save('tests/output/doctest-transforms-affine-rotate.png')

Offset          |  Scale         |  Rotate
:--------------:|:--------------:|:------------:
![Expected image](/tests/output/doctest-transforms-affine-offset.png) | ![Expected image](/tests/output/doctest-transforms-affine-scale.png) | ![Expected image](/tests/output/doctest-transforms-affine-rotate.png)

### *Polynomial transformations*

Polynomial transforms are typically performed using the 1st, 2nd, or 3rd order, with higher orders providing a tighter fit to the data:

    >>> for order in [1,2,3]:
    ...     trans = tio.transforms.Polynomial(order=order)
    ...     trans.fit(imx, imy, geox, geoy)
    ...     # warp the image
    ...     warped,affine = tio.imwarp.warp(im, trans)
    ...     warped.save('tests/output/doctest-transforms-polynomial-{}.png'.format(order))

1st order       |  2nd order     |  3rd order
:--------------:|:--------------:|:------------:
![Expected image](/tests/output/doctest-transforms-polynomial-1.png) | ![Expected image](/tests/output/doctest-transforms-polynomial-2.png) | ![Expected image](/tests/output/doctest-transforms-polynomial-3.png)

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

### *Chained transforms*

A particularly powerful feature is the `transformio.transforms.Chain` transform wrapper, which allows you to chain one or more transforms in a sequence. More... 

### *Custom transforms*

Implementing your own custom transform is easy. Simply write your own class with the following methods: 
- from_json: ...
- to_json: ...
- fit: ...
- predict: ...
- inverse: ... 


## Accuracy evaluation

An important but often overlooked aspect of coordinate transformation such as map georeferencing is the impact it has on accuracy or error. Transformio has key functionality for evaluating the accuracy of a transformation, so that the overall confidence in the final product can be evaluated. This functionality is contained in the `accuracy` module. 

### Within-sample model fit residual errors

To measure the model fit accuracy of a particular transform, we take the control points that we want to evaluate and measure the residual errors between the observed and predicted coordinates in either forward or backward direction. 

For instance, let's say we have a set of control points and want to know how well an Affine transform would predict the forward geographic coordinates. If the geographic coordinates are given in latitude-longitude coordinates, the errors are measured using geodesic distances, and can be summarized using one of several metrics:

    >>> trans = tio.transforms.Affine()
    >>> predicted,resids = tio.accuracy.residuals(trans, impoints, geopoints, distance='geodesic')

    >>> 'RMSE: {} km'.format(tio.accuracy.RMSE(resids)) # root mean square error
    'RMSE: 611.209296883062 km'

    >>> 'MAE: {} km'.format(tio.accuracy.MAE(resids)) # mean average error
    'MAE: 476.2463332384581 km'

    >>> 'MAX: {} km'.format(tio.accuracy.MAX(resids)) # maximum error
    'MAX: 1972.5992842092362 km'

    >>> for obs,pred,resid in zip(geopoints,predicted,resids):
    ...     'Observed {}; Predicted {}; Residual {} km'.format(obs,pred,resid)
    'Observed (101.621839, 56.161959); Predicted (98.6393264938272, 55.091630511199774); Residual 221.69318994970638 km'
    'Observed (71.44598, 51.1801); Predicted (74.59316421331444, 51.141083116350664); Residual 219.35642072534165 km'
    'Observed (80.26669, 50.42675); Predicted (81.66038500705729, 49.062378867594916); Residual 181.660439958306 km'
    'Observed (83.76361, 53.36056); Predicted (84.77938680649493, 52.02740211534941); Residual 163.1762934535337 km'
    'Observed (135.08379, 48.48272); Predicted (127.42165015818279, 51.63428657700119); Residual 648.6969056485497 km'
    'Observed (106.88324, 47.90771); Predicted (104.29233031094, 45.73824387979517); Residual 311.3013446554603 km'
    'Observed (68.04073, 33.12699); Predicted (75.81187709158846, 37.50926969298354); Residual 856.0361583885884 km'
    'Observed (119.70478, 31.94689); Predicted (120.01287943355501, 38.34193058379296); Residual 711.1994311343066 km'
    'Observed (129.04028, 35.10278); Predicted (128.44320342365774, 36.090794384765225); Residual 122.33204240889039 km'
    'Observed (94.900606, 36.406717); Predicted (93.68588357119687, 32.29957072259194); Residual 469.8017282067807 km'
    'Observed (121.05804, 29.32955); Predicted (124.84511290742313, 30.067238125275654); Residual 374.6176340935597 km'
    'Observed (78.715422, 21.426482); Predicted (75.3999079060918, 23.827998152433587); Residual 432.25107259271533 km'
    'Observed (91.1000101308, 29.6450238231); Predicted (90.2758097544798, 25.106870895335863); Residual 510.8130011737935 km'
    'Observed (96.86525, 21.09148); Predicted (95.29326437565582, 22.542103138695992); Residual 228.65743905727612 km'
    'Observed (121.7423789, 24.7184669); Predicted (121.79761020536966, 23.82102670605594); Residual 99.88497001009252 km'
    'Observed (121.56833333333, 25.03583333333); Predicted (124.44165076854335, 23.16305825090263); Residual 358.11836925811156 km'
    'Observed (118.080017048, 24.4499920847); Predicted (121.02008246244293, 21.873474216145553); Residual 414.94464704323633 km'
    'Observed (98.70707, 23.43771); Predicted (103.43905560010793, 20.222041777413985); Residual 604.8616827846928 km'
    'Observed (113.325010131, 23.1449813019); Predicted (115.91239894121175, 19.452915611489544); Residual 489.95457834099403 km'
    'Observed (111.2626075, 1.0875755); Predicted (96.30287917563957, 10.79685687946862); Residual 1972.5992842092362 km'
    'Observed (107.59546, 16.4619); Predicted (110.46798903128736, 11.107716981709608); Residual 670.8445533127542 km'
    'Observed (125.567222, 8.805556); Predicted (127.80251840498426, 11.81983575661313); Residual 414.61814484015173 km'

We can also do this in the backward direction and calculate the pixel sampling errors for each control point. To do so, we try to predict in the opposite direction, i.e. predicting the input pixel coordinates based on the output geographic coordinates. The errors are then measured as the eucliedian distance between the original and predicted input pixel coordinates: 

    >>> predicted,resids = tio.accuracy.residuals(trans, geopoints, impoints, distance='euclidean')

    >>> 'RMSE: {} pixels'.format(tio.accuracy.RMSE(resids))
    'RMSE: 102.14884780375236 pixels'

    >>> 'MAE: {} pixels'.format(tio.accuracy.MAE(resids))
    'MAE: 83.97566786748722 pixels'

    >>> 'MAX: {} pixels'.format(tio.accuracy.MAX(resids))
    'MAX: 282.60313208231486 pixels'

    >>> for obs,pred,resid in zip(impoints,predicted,resids):
    ...     'Observed {}; Predicted {}; Residual {} pixels'.format(obs,pred,resid)
    'Observed (532, 64); Predicted (591.4781711914964, 77.04904444945544); Residual 60.89277797349051 pixels'
    'Observed (113, 112); Predicted (92.71422935067517, 138.36979089162185); Residual 33.2697815217484 pixels'
    'Observed (230, 161); Predicted (233.97117489193306, 160.82567814611969); Residual 3.974999135982274 pixels'
    'Observed (289, 107); Predicted (296.60869896990744, 110.65453744457739); Residual 8.4408497172079 pixels'
    'Observed (1018, 166); Predicted (1117.4464383985583, 250.34757372576723); Residual 130.3997979430259 pixels'
    'Observed (611, 253); Predicted (659.710242442967, 233.07850407823537); Residual 52.62654956021256 pixels'
    'Observed (108, 379); Predicted (0.5173615544460972, 464.9375363425153); Residual 137.61459849826542 pixels'
    'Observed (866, 416); Predicted (834.5145394487631, 537.3928148933994); Residual 125.40952808246521 pixels'
    'Observed (1006, 470); Predicted (992.1323082955646, 488.91412195029955); Residual 23.453291503754212 pixels'
    'Observed (404, 502); Predicted (442.1019983917863, 431.45714748138187); Residual 80.17516026121304 pixels'
    'Observed (933, 583); Predicted (851.0492661778035, 586.5578495097557); Residual 82.02792858003043 pixels'
    'Observed (75, 645); Predicted (149.31246533392755, 689.2796986791675); Residual 86.5045329397486 pixels'
    'Observed (332, 638); Predicted (366.68854841347365, 551.2840130071816); Residual 93.39677612836839 pixels'
    'Observed (413, 694); Predicted (442.4630464124066, 713.2793864556681); Residual 35.21030880305688 pixels'
    'Observed (869, 701); Predicted (852.660810312578, 671.4998003086648); Residual 33.722854290083625 pixels'
    'Observed (913, 717); Predicted (850.4947097647739, 665.5284714653851); Residual 80.97054746683835 pixels'
    'Observed (852, 738); Predicted (792.8174030531827, 672.7987866126248); Residual 88.05553934055179 pixels'
    'Observed (548, 749); Predicted (477.0987283346137, 672.2160467927424); Residual 104.51203659819872 pixels'
    'Observed (760, 779); Predicted (713.1562617389629, 691.9642926744355); Residual 98.8410348181861 pixels'
    'Observed (408, 924); Predicted (634.4778010810271, 1093.033534780046); Residual 282.60313208231486 pixels'
    'Observed (651, 935); Predicted (606.6752262835507, 808.4545088766896); Residual 134.0837308872865 pixels'
    'Observed (949, 942); Predicted (881.9105701575786, 966.0768614342543); Residual 71.27893695268935 pixels'

### Leave-one-out model prediction residual errors

An important aspect to note about within-sample residuals and accuracy metrics is that they are not comparable across different transform model types, since each transform type has a different level of fit to the observed data. For instance, higher order polynomial transforms have a closer fit to the data and therefore have smaller residual errors than lower order ones. Map projection transforms are exact transforms without any loss of information and will therefore have zero residual errors. TIN or piecewise transforms have an exact fit at the control points and only contain errors inside the areas that are between the control points. 

    >>> # polynomial errors
    >>> for order in [1,2,3]:
    ...     trans = tio.transforms.Polynomial(order=order)
    ...     predicted,resids = tio.accuracy.residuals(trans, impoints, geopoints, distance='geodesic')
    ...     'RMSE: {:.9f} km'.format(tio.accuracy.RMSE(resids))
    'RMSE: 611.209296883 km'
    'RMSE: 436.849855103 km'
    'RMSE: 348.122455875 km'

    >>> # map projection error
    >>> trans = tio.transforms.MapProjection(fromcrs, tocrs)
    >>> projx,projy = trans.predict(*zip(*geopoints))
    >>> projpoints = list(zip(projx,projy))
    >>> predicted,resids = tio.accuracy.residuals(trans, geopoints, projpoints, distance='euclidean')

    >>> 'RMSE: {:.9f} m'.format(tio.accuracy.RMSE(resids))
    'RMSE: 0.000000000 m'

    >>> # TIN error
    >>> trans = tio.transforms.TIN()
    >>> predicted,resids = tio.accuracy.residuals(trans, impoints, geopoints, distance='geodesic')
    
    >>> 'RMSE: {:.9f} km'.format(tio.accuracy.RMSE(resids))
    'RMSE: 0.000000000 km'

Because of these problems with the traditional within-sample model residuals, a more comparable way of measuring accuracy is to instead use leave-one-out errors (also known as out-of-sample errors). The idea here is to calculate for each control point, where that point would be predicted if it was left out of the model. We do this by using the `loo_residuals` function, giving us a more comparable view of how the different models compare to each other: 

    >>> # polynomial errors
    >>> for order in [1,2,3]:
    ...     trans = tio.transforms.Polynomial(order=order)
    ...     predicted,resids = tio.accuracy.loo_residuals(trans, impoints, geopoints, distance='geodesic')
    ...     'RMSE: {:.9f} km'.format(tio.accuracy.RMSE(resids))
    'RMSE: 729.482755488 km'
    'RMSE: 684.036557435 km'
    'RMSE: 975.841451696 km'

    >>> # map projection error
    >>> trans = tio.transforms.MapProjection(fromcrs, tocrs)
    >>> projx,projy = trans.predict(*zip(*geopoints))
    >>> projpoints = list(zip(projx,projy))
    >>> predicted,resids = tio.accuracy.loo_residuals(trans, geopoints, projpoints, distance='euclidean')

    >>> 'RMSE: {:.9f} km'.format(tio.accuracy.RMSE(resids))
    'RMSE: 0.000000000 km'

    >>> # TIN error
    >>> trans = tio.transforms.TIN()
    >>> predicted,resids = tio.accuracy.loo_residuals(trans, impoints, geopoints, distance='geodesic')

    >>> 'RMSE: {:.9f} km'.format(tio.accuracy.RMSE(resids))
    'RMSE: 688.720731518 km'


## Control point outliers

The ground control points used to fit transformation models are often noisy and contain outliers, resulting in an adverse effect on the fitted transform. Therefore, we need efficient and automated ways to detect and exclude these. Transformio includes a set of functions to help with this. Let's start with the full set of 22 control points for an affine transform: 

    >>> # get the error associated with all control points
    >>> trans = tio.transforms.Affine()
    >>> predicted,resids = tio.accuracy.residuals(trans, impoints, geopoints, distance='geodesic')

    >>> 'Control points: {}'.format(len(impoints))
    'Control points: 22'

    >>> 'RMSE: {:.9f} km'.format(tio.accuracy.RMSE(resids))
    'RMSE: 611.209296883 km'

    >>> # warp the image
    >>> trans.fit(*zip(*impoints), *zip(*geopoints))
    >>> warped,affine = tio.imwarp.warp(im, trans)
    >>> bounds = tio.imwarp.imbounds(*warped.size, trans)

    >>> # visualize on the map
    >>> draw = warped.copy()
    >>> tio.utils.draw_geojson({'type':'MultiPoint', 'coordinates':predicted},
    ...                          draw, bounds, fillcolor="red", fillsize=25)
    >>> tio.utils.draw_geojson({'type':'MultiPoint', 'coordinates':geopoints},
    ...                          draw, bounds, fillcolor="green", fillsize=25)
    >>> draw.save('tests/output/doctest-outliers-all.png')

![Outliers all](/tests/output/doctest-outliers-all.png)

To improve this accuracy we can compare the different control points and drop the one whose exclusion best improves the accuracy metric. We can do this continually until the model stops improving beyond a given threshold: 

    >>> # continually drop the control point with the worst performing model
    >>> _trans, _impoints, _geopoints, _predicted, _resids, _err = tio.accuracy.auto_drop_models(trans, impoints, geopoints, distance='geodesic', metric='rmse', improvement_ratio=0.10)

    >>> 'Control points: {}'.format(len(_impoints))
    'Control points: 19'

    >>> 'RMSE: {:.9f} km'.format(_err)
    'RMSE: 333.882194848 km'

    >>> # warp the image
    >>> _trans.fit(*zip(*_impoints), *zip(*_geopoints))
    >>> warped,affine = tio.imwarp.warp(im, _trans)
    >>> bounds = tio.imwarp.imbounds(*warped.size, _trans)

    >>> # visualize on the map
    >>> draw = warped.copy()
    >>> tio.utils.draw_geojson({'type':'MultiPoint', 'coordinates':_predicted},
    ...                          draw, bounds, fillcolor="red", fillsize=25)
    >>> tio.utils.draw_geojson({'type':'MultiPoint', 'coordinates':_geopoints},
    ...                          draw, bounds, fillcolor="green", fillsize=25)
    >>> draw.save('tests/output/doctest-outliers-dropped.png')

![Outliers dropped](/tests/output/doctest-outliers-dropped.png)


## Model selection

In addition to evaluating individual models, transformio provides functionality for automatically finding the optimal set of control points for multiple transformation models, and then selecting the most accurate model based on leave-one-out residuals: 

    >>> # auto choose from a selection of models
    >>> trytransforms = [tio.transforms.Polynomial(order=1),
    ...                 tio.transforms.Polynomial(order=2),
    ...                 tio.transforms.Polynomial(order=3),
    ...                 tio.transforms.TIN()]
    >>> _trans, _impoints, _geopoints, _predicted, _resids, _err = tio.accuracy.auto_choose_model(impoints, geopoints, trytransforms, refine_outliers=True, distance='geodesic', metric='rmse')

    >>> _trans
    Polynomial Transform(order=2, estimated=True)

    >>> 'Control points: {}'.format(len(_impoints))
    'Control points: 18'

    >>> 'RMSE: {:.9f} km'.format(_err)
    'RMSE: 278.617602953 km'

    >>> # warp the image
    >>> _trans.fit(*zip(*_impoints), *zip(*_geopoints))
    >>> warped,affine = tio.imwarp.warp(im, _trans)
    >>> bounds = tio.imwarp.imbounds(*warped.size, _trans)

    >>> # visualize on the map
    >>> draw = warped.copy()
    >>> tio.utils.draw_geojson({'type':'MultiPoint', 'coordinates':_predicted},
    ...                          draw, bounds, fillcolor="red", fillsize=25)
    >>> tio.utils.draw_geojson({'type':'MultiPoint', 'coordinates':_geopoints},
    ...                          draw, bounds, fillcolor="green", fillsize=25)
    >>> draw.save('tests/output/doctest-automodel.png')
    
![Outliers dropped](/tests/output/doctest-automodel.png)


## Vector examples

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

### *Simple geometry manipulations*

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

### *Systematic bias correction*

One of the situations where this becomes useful, is when needing to make adjusts or corrections to a dataset. For instance, if we have the boundary data for a subnational administrative boundary, but notice through visual inspection that it is systematically biased or offset in the southward and westward direction when compared to some reference data. This type of error can often happen if the underlaying map data which was used to digitize the data was georeferenced incorrectly. We can then easily offset this bias by using an Affine transform:

    >>> # first load the data
    >>> data = json.loads(open('tests/data/ARM-ADM1-Lori-Natural_Earth.geojson').read())['geometry']
    >>> reference = json.loads(open('tests/data/ARM-ADM1-Lori-IPUMS.geojson').read())['geometry']

    >>> # visualize the original bias
    >>> im,bounds = tio.utils.draw_geojson(reference, fillcolor=(0,0,255,120), outlinecolor=(0,0,255), outlinewidth=3)
    >>> tio.utils.draw_geojson(data, im, bounds, fillcolor=None, outlinecolor=(255,0,0), outlinewidth=3)
    >>> im.save('tests/output/doctest-vector-bias-orig.png')

    >>> # correct the bias
    >>> trans = tio.transforms.Affine(offset=(0.025,0.025))
    >>> data = tio.vector.transform(data, trans)

    >>> # visualize the correction
    >>> im,bounds = tio.utils.draw_geojson(reference, fillcolor=(0,0,255,120), outlinecolor=(0,0,255), outlinewidth=3)
    >>> tio.utils.draw_geojson(data, im, bounds, fillcolor=None, outlinecolor=(255,0,0), outlinewidth=3)
    >>> im.save('tests/output/doctest-vector-bias-corrected.png')

Biased                   |  Corrected
:-------------------------:|:-------------------------:
![Original image](/tests/output/doctest-vector-bias-orig.png) | ![Expected image](/tests/output/doctest-vector-bias-corrected.png)

### *Digitized geometry transformation*

Another example is when digitizing the data contents of a scanned map image or remote sensing imagery. If the digitized geometry coordinates were recorded in image space using pixel coordinates, they would need to be transformed to geographic space based on the georeferencing information of the source imagery. For instance, if the source image coordinate system was georeferenced using a 2nd order polynomial function, the extracted vector geometries are transformed as follows:

    >>> # first load the digitized data
    >>> digitized = json.loads(open('tests/data/argentina_pol96_digitized.geojson').read())

    >>> # visualize the digitized data in image space
    >>> im = Image.open('tests/data/argentina_pol96.jpg')
    >>> w,h = im.size
    >>> imbounds = [0,0,w,h]
    >>> draw = im.copy()
    >>> tio.utils.draw_geojson(digitized, draw, imbounds, fillcolor=(0,0,255,120), outlinecolor=(0,0,255), outlinewidth=5)
    >>> draw.save('tests/output/doctest-vector-digitized-orig.png')

    >>> # next load the transform parameters determined through georeferencing
    >>> with open('tests/data/argentina_pol96_georeferenced_transform.json') as r:
    ...     transinfo = json.loads(r.read())
    >>> forw = transinfo['forward']['data']['A']
    >>> back = transinfo['backward']['data']['A']
    >>> trans = tio.transforms.Polynomial(A=forw, Ainv=back)

    >>> # transform the digitized data to geographic space
    >>> digitized = tio.vector.transform(digitized, trans)

    >>> # visualize the digitized data in transformed geographic space
    >>> warped,affine = tio.imwarp.warp(im, trans)
    >>> bounds = tio.imwarp.imbounds(*im.size, trans)
    >>> draw = warped.copy()
    >>> tio.utils.draw_geojson(digitized, draw, bounds, fillcolor=(0,0,255,120), outlinecolor=(0,0,255), outlinewidth=5)
    >>> draw.save('tests/output/doctest-vector-digitized-transformed.png')

Image space                   |  Geographic space
:-------------------------:|:-------------------------:
![Original image](/tests/output/doctest-vector-digitized-orig.png) | ![Expected image](/tests/output/doctest-vector-digitized-transformed.png)


## Raster examples

For raster datasets which consist of regularly spaced grid cells, each cell or image pixel needs to be projected and resampled to another coordinate system. Such raster transformation is more complicated and involves more steps than in the case of vector transformation. Transformio greatly simplifies this process, and works seamlessly with the Pillow imaging library. 

There are two basic steps to this process that transformio handles behind the scenes. 

1. The first is the process of mapping coordinates between the two coordinate systems. Forward or backward... 

2. The second is how to resample and interpolate the pixels. Currently, transformio only supports nearest neighbour resampling... 

### *Raster reprojection*

One common use-case is just reprojecting a raster dataset from one projection to another. Assume we have the following latitude-longitude WGS84 satellite image that we know covers the entire globe: 

![Original image](/tests/data/land_shallow_topo_2048.png)

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

That's all we need. Now just call on the `warp` function along with a list specifying that these two transforms should be called in sequence - first from image to source geographic space then from source to target geographic space: 

    >>> # warp the image
    >>> warped,affine = tio.imwarp.warp(im, [img2geo,geo2proj])
    >>> warped.save('tests/output/doctest-raster-reprojection.png')

![Expected image](/tests/output/doctest-raster-reprojection.png)

This returns a PIL image containing the image contents after warping it into the output coordinate system, and the affine transform parameters defining this coordinate system. The output image size is automatically determined to approximately match the input image size, but can also be set manually using the `size` arg. Finally, you can also define which area in the output coordinate system to include in the output image using the `bounds` arg, which can be useful for tiled reprojection. For instance:

    >>> # warp the image
    >>> warped,affine = tio.imwarp.warp(im, [img2geo,geo2proj], bounds=[0,0,18038694,9020047], size=(256,256))
    >>> warped.save('tests/output/doctest-raster-reprojection-tile.png')

![Expected image](/tests/output/doctest-raster-reprojection-tile.png)



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

Remote sensing imagery from satellites or aircraft similarly needs to undergo a process of registration or rectification that adjusts and defines the coordinates of the image. This can be done either manually by matching distinct point features, or automatically based on pixel region correlations with known imagery. The identified control points are then used to fit a transformation function and warp the image in the same way as before:

    >>> # first load image
    >>> im = Image.open('tests/data/satim-volcano.jpg')
    >>> impoints = [(361,1814),(834,845),(1490,1688),(1925,1461),(1365,916)]
    >>> geopoints = [(158.5327148376,53.0651510535),(158.7098693793,53.3199366405),(159.0422058051,53.097322592),(159.2344665472,53.1582999509),(158.9859008734,53.3046210736)]

    >>> # visualize on a map
    >>> geoj = {'type':'MultiPoint', 'coordinates':impoints}
    >>> w,h = im.size
    >>> imbounds = [0,0,w,h]
    >>> draw = im.copy()
    >>> tio.utils.draw_geojson(geoj, draw, imbounds, fillcolor="red", fillsize=15)
    >>> draw.save('tests/output/doctest-satim-controlpoints.png')

![Landsat 8 satellite image of Petropavlovsk-Kamchatsky. Source URL: https://innoter.com/upload/iblock/33f/20130824-%D0%9F%D0%B5%D1%82%D1%80%D0%BE%D0%BF%D0%B0%D0%B2%D0%BB%D0%BE%D0%B2%D1%81%D0%BA-%D0%9A%D0%B0%D0%BC%D1%87%D0%B0%D1%82%D1%81%D0%BA%D0%B8%D0%B9.jpg](/tests/output/doctest-satim-controlpoints.png)

    >>> # create and fit the transform model
    >>> imx,imy = zip(*impoints)
    >>> geox,geoy = zip(*geopoints)
    >>> trans = tio.transforms.Polynomial()
    >>> trans.fit(imx, imy, geox, geoy)
    Polynomial Transform(order=1, estimated=True)

    >>> # warp the image
    >>> warped,affine = tio.imwarp.warp(im, trans)

    >>> # visualize on the map
    >>> geoj = {'type':'MultiPoint', 'coordinates':geopoints}
    >>> bounds = tio.imwarp.imbounds(*warped.size, trans)
    >>> tio.utils.draw_geojson(geoj, warped, bounds, fillcolor="red", fillsize=15)
    >>> warped.save('tests/output/doctest-satim-georeferenced.png')

![Expected image](/tests/output/doctest-satim-georeferenced.png)

More advanced cases use information about elevation and camera perspective to correct various distortions, but this is not yet supported here. 


