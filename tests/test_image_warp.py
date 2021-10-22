
import unittest
from numpy.testing import assert_array_almost_equal
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

class TestImbounds(unittest.TestCase):
    
    def setUp(self):
        # define the transform for a raster with a cell size of 1x1 degrees
        # covering the entire world
        self.width,self.height = 360,180
        self.trans = tio.transforms.Affine(offset=(-180,90), scale=(1,-1))

    def test_topleft(self):
        # check that the coordinate of topleft pixel is at -180,90
        xpred,ypred = self.trans.predict([0], [0])
        assert_array_almost_equal(xpred, [-180])
        assert_array_almost_equal(ypred, [90])

    def test_bottomright(self):
        # check that the coordinate of bottomright pixel is at 179,-89
        xpred,ypred = self.trans.predict([self.width-1], [self.height-1])
        assert_array_almost_equal(xpred, [179])
        assert_array_almost_equal(ypred, [-89])

    def test_bottomright_plus(self):
        # check that the coordinate of bottomright+1 pixel is at 180,-90
        xpred,ypred = self.trans.predict([self.width], [self.height])
        assert_array_almost_equal(xpred, [180])
        assert_array_almost_equal(ypred, [-90])

    def test_correct_imbounds(self):
        # check that imbounds correctly returns -180,90,180,-90
        bounds = tio.imwarp.imbounds(self.width, self.height, self.trans)
        assert_array_almost_equal(bounds, [-180,-90,180,90])

    def test_imbounds_fitbounds(self):
        # check that using imbounds as input to fitbounds reproduces imbounds
        bounds = tio.imwarp.imbounds(self.width, self.height, self.trans)
        btrans = tio.imwarp.fitbounds(self.width, self.height, bounds)
        (bx1,bx2),(by1,by2) = btrans.predict([0,self.width], [0,self.height])
        assert_array_almost_equal(bounds, [bx1,by1,bx2,by2])


if __name__ == '__main__':
    unittest.main()




