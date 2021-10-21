
import unittest
import transformio as tio

class BaseTestCases(object):

    class TestWorldSatellite(unittest.TestCase):
        fromcrs = '+proj=longlat +datum=WGS84 +no_defs ' #'epsg:4326'

        def setUp(self):
            # load a satellite image and its affine transform
            from PIL import Image
            self.im = Image.open('data/land_shallow_topo_2048.tif')
            self.bounds = [-180,90,180,-90] # left,upper,right,bottom
            self.affine = tio.imwarp.fitbounds(self.im.size[0], self.im.size[1], self.bounds)

        def test_reproject(self):
            # define target projection transform
            proj = tio.transforms.MapProjection(self.fromcrs, self.tocrs)
            # warp the image
            warped,warped_affine = tio.imwarp.warp(self.im, [self.affine,proj])
            warped.save('output/test_image_projections_{}.png'.format(self.name))

class TestWorldRobinson(BaseTestCases.TestWorldSatellite):
    name = 'robinson'
    tocrs = 'esri:54030'

class TestWorldMollweide(BaseTestCases.TestWorldSatellite):
    name = 'mollweide'
    tocrs = 'esri:54009'

class TestWorldVanDerGrinten(BaseTestCases.TestWorldSatellite):
    name = 'vandergrinten'
    tocrs = 'esri:54029'

class TestWorldArcticPolar(BaseTestCases.TestWorldSatellite):
    name = 'arcticpolar'
    tocrs = 'epsg:3995'

class TestWorldArcticPolar(BaseTestCases.TestWorldSatellite):
    name = 'antarcticpolar'
    tocrs = 'epsg:3031'

class TestWorldNorwayPolar(BaseTestCases.TestWorldSatellite):
    name = 'norwaypolar'
    tocrs = 'epsg:5939'

if __name__ == '__main__':
    unittest.main()
