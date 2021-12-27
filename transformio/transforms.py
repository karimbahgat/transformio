
import numpy as np
import math

# important note about noninvertible polynomial transforms
# https://blogs.mathworks.com/steve/2006/07/27/spatial-transformations-handling-noninvertible-cases/

model_dict_doc = '''
    - type: the name of the transformation. 
    - params: a dict of parameters defining a transform model. 
    - data: a dict of parameters representing the data of a specific fitted transform model. 
    '''

to_json_doc = '''Returns a JSON compatible dict sufficient to store and recreate the transformation.
    The returned dict has the following structure:
    {}
    '''.format(model_dict_doc)

def from_json(js):
    '''Creates and returns a transform object from a transform dict definition
    as returned by the corresponding transform object's `info()` method.

    The dict should have the following structure:
    {}
    '''.format(model_dict_doc)
    cls = {#'Similarity':Similarity,
           'Affine':Affine,
           #'Projective':Projective,
           'Polynomial':Polynomial,
           'MapProjection':MapProjection,
           'TIN':TIN,
           'Chain':Chain,
           }[js['type']]
    trans = cls.from_json(js)
    return trans

class Chain(object):

    def __init__(self, transforms=None):
        '''A chain of multiple transforms executed consecutively'''
        self.transforms = [t for t in transforms] if transforms else []

    def __repr__(self):
        return u'Chain Transform(transforms={})'.format(self.transforms)

    def copy(self):
        transforms = [t.copy() for t in self.transforms]
        new = Chain(transforms)
        return new

    def info(self):
        '''For backward-compatibility. See instead `.to_json()`'''
        return self.to_json()
    
    def to_json(self):
        '''{}'''.format(to_json_doc)
        params = {}
        data = {'transforms':[trans.info() for trans in self.transforms] }
        info = {'type': 'Chain',
                'params': params,
                'data': data,
                }
        return info

    @staticmethod
    def from_json(js):
        init = {}
        transforms = [from_json(transdict) for transdict in js['data']['transforms']]
        init['transforms'] = transforms
        init.update(js['params'])
        trans = Chain(**init)
        return trans

    def add(self, transform):
        self.transforms.append(transform)

    def inverse(self):
        transforms = [tr.inverse() for tr in reversed(self.transforms)]
        inv = Chain(transforms)
        return inv

    def predict(self, x, y):
        for trans in self.transforms:
            x,y = trans.predict(x, y)
        return x,y

class Polynomial(object):
    
    def __init__(self, order=None, A=None, Ainv=None):
        '''Polynomial transform of 1st (affine), 2nd, or 3rd order'''
        if A is not None:
            A = np.array(A)
            if A.shape == (3,3):
                order = 1
            elif A.shape == (6,6):
                order = 2
            elif A.shape == (10,10):
                order = 3
            else:
                raise ValueError('Matrix A must be shape (3,3), (6,6), or (10,10); not {}'.format(A.shape))

        if Ainv is not None:
            Ainv = np.array(Ainv)
            assert Ainv.shape == A.shape

        self.A = A
        self.Ainv = Ainv
        self.order = order
        self.minpoints = {1:3, 2:10, 3:20}.get(order, 3) # minimum 3 if order not set

    def __repr__(self):
        return u'Polynomial Transform(order={}, estimated={})'.format(self.order, self.A is not None)

    def copy(self):
        new = Polynomial(order=self.order, A=self.A, Ainv=self.Ainv)
        new.minpoints = self.minpoints
        return new

    def info(self):
        '''For backward-compatibility. See instead `.to_json()`'''
        return self.to_json()
    
    def to_json(self):
        '''{}'''.format(to_json_doc)
        params = {'order': self.order}
        data = {'A': self.A.tolist() }
        if self.Ainv is not None:
            data['Ainv'] = self.Ainv.tolist()
        info = {'type': 'Polynomial',
                'params': params,
                'data': data,
                }
        return info

    @staticmethod
    def from_json(js):
        init = {}
        A = np.array(js['data']['A'])
        init['A'] = A
        if 'Ainv' in js['data']:
            Ainv = np.array(js['data']['Ainv'])
            init['Ainv'] = Ainv
        init.update(js['params'])
        trans = Polynomial(**init)
        return trans

    def fit(self, inx, iny, outx, outy, invert=True): 
        # to arrays
        inx = np.array(inx)
        iny = np.array(iny)
        outx = np.array(outx)
        outy = np.array(outy)

        # auto determine order from number of points
        if not self.order:
            # due to automation and high likelihood of errors, we set higher point threshold for polynomial order
            # compare to gdal: https://github.com/naturalatlas/node-gdal/blob/master/deps/libgdal/gdal/alg/gdal_crs.c#L186
            if len(inx) >= 20:
                self.order = 3
            if len(inx) >= 10:
                self.order = 2
            else:
                self.order = 1
            # update minpoints
            self.minpoints = {1:3, 2:10, 3:20}[self.order] 
        
        if self.order == 1:
            # terms
            x = inx
            y = iny
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([x,y,ones]).transpose()
            # x and y coeffs
            xcoeffs,xres,xrank,xsing = np.linalg.lstsq(u, outx, rcond=-1) 
            ycoeffs,yres,yrank,ysing = np.linalg.lstsq(u, outy, rcond=-1)
            # A matrix
            A = np.eye(3)
            # two first rows of the A matrix are equations for the x and y coordinates, respectively
            A[0,:] = xcoeffs
            A[1,:] = ycoeffs
            
        elif self.order == 2:
            
            # inverse
            if invert:
                # ALT1
                # standard switch the points
                backward = self.copy()
                backward.fit(outx, outy, inx, iny, invert=False)
                Ainv = backward.A

                # ALT2
                # fit the forward transform
##                forward = self.copy()
##                forward.fit(inx, iny, outx, outy)
##                
##                # forward predict the gcps
##                x_pred,y_pred = forward.predict(inx, iny)
##
##                # get backward transform by fitting the forward predicted gcps to the input gcps
##                backward = self.copy()
##                backward.fit(x_pred, y_pred, inx, iny)
##                Ainv = backward.A

                # ALT3
##                # fit the forward transform
##                forward = self.copy()
##                forward.fit(inx, iny, outx, outy)
##                
##                # forward predict regularly spaced sample points across the range of the inpoints
##                xmin,ymin,xmax,ymax = inx.min(), iny.min(), inx.max(), iny.max()
##                x = np.linspace(xmin, xmax, 100)
##                y = np.linspace(ymin, ymax, 100)
##                x,y = np.meshgrid(x, y)
##                x,y = x.flatten(), y.flatten()
##                x_pred,y_pred = forward.predict(x, y)
##
##                # get backward transform by fitting the forward predicted sample points to the sample points
##                # should be a near perfect match (~0 residuals) since these are derived from the same transform
##                backward = self.copy()
##                backward.fit(x_pred, y_pred, x, y)
##                Ainv = backward.A

            # forward
            # terms
            x = inx
            y = iny
            xx = x*x
            xy = x*y
            yy = y*y
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([xx,xy,yy,x,y,ones]).transpose()
            # find best coefficients for all equivalent points using least squares
            xcoeffs,xres,xrank,xsing = np.linalg.lstsq(u, outx, rcond=-1) 
            ycoeffs,yres,yrank,ysing = np.linalg.lstsq(u, outy, rcond=-1)
            # A matrix
            A = np.eye(6)
            # two first rows of the A matrix are equations for the x and y coordinates, respectively
            A[0,:] = xcoeffs
            A[1,:] = ycoeffs

        elif self.order == 3:
            
            # inverse
            if invert:
                # ALT1
                # standard switch the points
                backward = self.copy()
                backward.fit(outx, outy, inx, iny, invert=False)
                Ainv = backward.A

                # ALT2
##                # fit the forward transform
##                forward = self.copy()
##                forward.fit(inx, iny, outx, outy)
##                
##                # forward predict the gcps
##                x_pred,y_pred = forward.predict(inx, iny)
##
##                # get backward transform by fitting the forward predicted gcps to the input gcps
##                backward = self.copy()
##                backward.fit(x_pred, y_pred, inx, iny)
##                Ainv = backward.A

                # ALT3
##                # fit the forward transform
##                forward = self.copy()
##                forward.fit(inx, iny, outx, outy)
##                
##                # forward predict regularly spaced sample points across the range of the inpoints
##                xmin,ymin,xmax,ymax = inx.min(), iny.min(), inx.max(), iny.max()
##                x = np.linspace(xmin, xmax, 100)
##                y = np.linspace(ymin, ymax, 100)
##                x,y = np.meshgrid(x, y)
##                x,y = x.flatten(), y.flatten()
##                x_pred,y_pred = forward.predict(x, y)
##
##                # get backward transform by fitting the forward predicted sample points to the sample points
##                # should be a near perfect match (~0 residuals) since these are derived from the same transform
##                backward = self.copy()
##                backward.fit(x_pred, y_pred, x, y)
##                Ainv = backward.A

            # forward
            # terms
            #X = a0 + a1x + a2y + a3xy + a4x^2 + a5y^2 + a6x^3 + a7x^2y + a8xy^2 + a9y^3
            #Y = b0 + b1x + b2y + b3xy + b4x^2 + b5y^2 + b6x^3 + b7x^2y + b8xy^2 + b9y^3
            x = inx
            y = iny
            xx = x*x
            xy = x*y
            yy = y*y
            xxx = xx*x
            xxy = xx*y
            xyy = x*yy
            yyy = yy*y
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([xxx,xxy,xyy,yyy, xx,xy,yy, x,y,ones]).transpose()
            # x and y coeffs
            xcoeffs,xres,xrank,xsing = np.linalg.lstsq(u, outx, rcond=-1) 
            ycoeffs,yres,yrank,ysing = np.linalg.lstsq(u, outy, rcond=-1)
            # A matrix
            A = np.eye(10)
            # two first rows of the A matrix are equations for the x and y coordinates, respectively
            A[0,:] = xcoeffs
            A[1,:] = ycoeffs

        self.A = A
        if self.order > 1 and invert:
            self.Ainv = Ainv
        return self

    def inverse(self):
        if self.order == 1:
            # calc Ainv and use as A
            Ainv = np.linalg.inv(self.A)
            inv = Polynomial(A=Ainv)
        elif self.order > 1:
            # flip Ainv and A
            inv = Polynomial(A=self.Ainv, Ainv=self.A)
        return inv

    def predict(self, x, y):
        # to arrays
        x = np.array(x)
        y = np.array(y)

        # input
        u = np.array([x,y])
        
        if self.order == 1:
            # terms
            x = x
            y = y
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([x,y,ones])
            
        elif self.order == 2:
            # terms
            x = x
            y = y
            xx = x*x
            xy = x*y
            yy = y*y
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([xx,xy,yy,x,y,ones])

        elif self.order == 3:
            # terms
            x = x
            y = y
            xx = x*x
            xy = x*y
            yy = y*y
            xxx = xx*x
            xxy = xx*y
            xyy = x*yy
            yyy = yy*y
            ones = np.ones(x.shape)
            # u consists of each term in equation, with each term being array if want to transform multiple
            u = np.array([xxx,xxy,xyy,yyy, xx,xy,yy, x,y,ones])

        # apply the transform matrix to predict output
        predx,predy = self.A.dot(u)[:2]
        return predx,predy



class Affine(object):
    
    def __init__(self, 
                scale=None, offset=None, rotate=None, skew=None,
                A=None):
        '''Affine transform, also known as a 1st order polynomial.'''
        if A is not None:
            A = np.array(A)
            if A.shape != (3,3):
                raise ValueError('Matrix A must be shape (3,3); not {}'.format(A.shape))

        else:
            A = np.eye(3)
            if scale:
                if isinstance(scale, tuple):
                    xscale,yscale = scale
                else:
                    xscale = yscale = scale
                A[0,0] = xscale
                A[1,1] = yscale
            if offset:
                if isinstance(offset, tuple):
                    xoff,yoff = offset
                else:
                    xoff = yoff = offset
                A[0,2] = xoff
                A[1,2] = yoff
            if rotate:
                angle = rotate
                A[0,0] = math.cos(angle)
                A[0,1] = -math.sin(angle)
                A[1,0] = math.sin(angle)
                A[1,1] = math.cos(angle)
            if skew:
                raise NotImplementedError()

        self.A = A
        self.minpoints = 3

    def __repr__(self):
        return u'Affine Transform(estimated={})'.format(self.A is not None)

    def copy(self):
        new = Affine(A=self.A)
        return new

    def info(self):
        '''For backward-compatibility. See instead `.to_json()`'''
        return self.to_json()
    
    def to_json(self):
        '''{}'''.format(to_json_doc)
        params = {}
        data = {'A': self.A.tolist() }
        info = {'type': 'Affine',
                'params': params,
                'data': data,
                }
        return info

    @staticmethod
    def from_json(js):
        init = {}
        A = np.array(js['data']['A'])
        init['A'] = A
        init.update(js['params'])
        trans = Affine(**init)
        return trans

    def fit(self, inx, iny, outx, outy, invert=False): #, exact=False):
        # to arrays
        inx = np.array(inx)
        iny = np.array(iny)
        outx = np.array(outx)
        outy = np.array(outy)
        
        # terms
        x = inx
        y = iny
        ones = np.ones(x.shape)
        # u consists of each term in equation, with each term being array if want to transform multiple
        u = np.array([x,y,ones]).transpose()
        # x and y coeffs
        xcoeffs,xres,xrank,xsing = np.linalg.lstsq(u, outx, rcond=-1) 
        ycoeffs,yres,yrank,ysing = np.linalg.lstsq(u, outy, rcond=-1)
        # A matrix
        A = np.eye(3)
        # two first rows of the A matrix are equations for the x and y coordinates, respectively
        A[0,:] = xcoeffs
        A[1,:] = ycoeffs

        self.A = A
        return self

    def inverse(self):
        Ainv = np.linalg.inv(self.A)
        inv = Affine(A=Ainv)
        return inv

    def predict(self, x, y):
        # to arrays
        x = np.array(x)
        y = np.array(y)

        # input
        u = np.array([x,y])
        
        # terms
        x = x
        y = y
        ones = np.ones(x.shape)
        # u consists of each term in equation, with each term being array if want to transform multiple
        u = np.array([x,y,ones])

        # apply the transform matrix to predict output
        predx,predy = self.A.dot(u)[:2]
        return predx,predy



class Projective(object):
    def __init__(self, fromcrs, tocrs):
        '''A perspective-like transform'''
        # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-08-TwoViews-1.pdf
        # https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20172018/LectureNotes/MATH/homogenous.html
        # https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20172018/LectureNotes/MATH/homogenous.html
        pass



class Similarity(object):
    def __init__(self, scale=None, offset=None, rotate=None,
                A=None):
        '''A similarity transform.
        
        This is a special case of the affine transform that only retains 
        uniform scaling, translation, rotation, or reflection.
        '''
        # https://scikit-image.org/docs/dev/api/skimage.transform.html#similaritytransform
        pass



class MapProjection(object):
    def __init__(self, fromcrs, tocrs):
        '''Map projection transform'''
        self.fromcrs = fromcrs
        self.tocrs = tocrs
        self.minpoints = 0

        import pyproj
        self._transformer = pyproj.Transformer.from_crs(fromcrs, tocrs)

    def __repr__(self):
        return u'Map Projection Transform(fromcrs={}, tocrs={})'.format(self.fromcrs, self.tocrs)

    def copy(self):
        new = MapProjection(fromcrs=self.fromcrs, tocrs=self.tocrs)
        new.minpoints = self.minpoints
        return new

    def info(self):
        '''For backward-compatibility. See instead `.to_json()`'''
        return self.to_json()
    
    def to_json(self):
        '''{}'''.format(to_json_doc)
        params = {}
        data = {'fromcrs': self.fromcrs,
                'tocrs': self.tocrs}
        info = {'type': 'MapProjection',
                'params': params,
                'data': data,
                }
        return info

    @staticmethod
    def from_json(js):
        init = {}
        init['fromcrs'] = js['data']['fromcrs']
        init['tocrs'] = js['data']['tocrs']
        trans = MapProjection(**init)
        return trans

    def fit(self, *args, **kwargs):
        # Note: The map projection transform is an analytic transformation and does not need to be fit or estimated
        return self

    def inverse(self):
        inv = MapProjection(self.tocrs, self.fromcrs)
        return inv

    def predict(self, x, y):
        predx,predy = self._transformer.transform(x, y)
        return np.array(predx),np.array(predy)



class TIN(object):
    def __init__(self, tris=None):
        '''Creates a triangulated irregular network (TIN) between control points
        and does a global affine transform within each triangle'''
        self.tris = tris or []
        self.minpoints = 3 # at least one triangle/affine

    def __repr__(self):
        return u'TIN Transform(estimated={})'.format(bool(self.tris))

    def copy(self):
        new = TIN()
        new.tris = list(self.tris)
        new.minpoints = self.minpoints
        return new

    def info(self):
        '''For backward-compatibility. See instead `.to_json()`'''
        return self.to_json()
    
    def to_json(self):
        '''{}'''.format(to_json_doc)
        params = {}
        tri_models = [(tri,trans.info()) for tri,trans in self.tris]
        data = {'tris': tri_models}
        info = {'type': 'TIN',
                'params': params,
                'data': data,
                }
        return info

    @staticmethod
    def from_json(js):
        init = {}
        init['tris'] = js['data']['tris']
        trans = TIN(**init)
        return trans

    def fit(self, inx, iny, outx, outy, invert=False):
        # to arrays
        inx = np.array(inx)
        iny = np.array(iny)
        outx = np.array(outx)
        outy = np.array(outy)

        # add corner coordinates to the input points
        inxmin,inymin,inxmax,inymax = inx.min(),iny.min(),inx.max(),iny.max()
        outxmin,outymin,outxmax,outymax = outx.min(),outy.min(),outx.max(),outy.max()
        # need to account for axis orientation? 
        inx1,iny1,inx2,iny2 = inxmin,inymin,inxmax,inymax
        outx1,outy1,outx2,outy2 = outxmin,outymin,outxmax,outymax
        # define and add the corners
        incorners = [(inx1,iny1),(inx2,iny1),(inx2,iny2),(inx1,iny2)]
        outcorners = [(outx1,outy1),(outx2,outy1),(outx2,outy2),(outx1,outy2)]
        inx_corners,iny_corners = zip(*incorners)
        outx_corners,outy_corners = zip(*outcorners)
        inx = np.append(inx, inx_corners)
        iny = np.append(iny, iny_corners)
        outx = np.append(outx, outx_corners)
        outy = np.append(outy, outy_corners)

        import shapely, shapely.geometry, shapely.ops

        inpoints = list(zip(inx,iny))
        inpoints = shapely.geometry.MultiPoint(inpoints)
        intris = shapely.ops.triangulate(inpoints)

        self.tris = []
        for intri in intris:
            intri_points = list(intri.exterior.coords)[:3]
            intri_x,intri_y = zip(*intri_points)
            outtri_x = [outx[inx==_x][0] for _x in intri_x]
            outtri_y = [outy[iny==_y][0] for _y in intri_y]
            outtri_points = list(zip(outtri_x, outtri_y))
            trans = Polynomial(1)
            trans.fit(intri_x, intri_y, outtri_x, outtri_y)
            self.tris.append((intri_points, trans))
        
        return self

    def inverse(self):
        invtris = []
        for tri,trans in self.tris:
            tri_x,tri_y = zip(*tri)
            tri_xpred,tri_ypred = trans.predict(tri_x, tri_y)
            tripred = zip(tri_xpred, tri_ypred)
            invtris.append( (tripred,trans.inverse()) )
        inv = TIN(invtris)
        return inv

    def predict(self, x, y):

        def point_in_tri(x1,y1,x2,y2,x3,y3,xp,yp):
            # https://www.w3resource.com/python-exercises/basic/python-basic-1-exercise-40.php
            # modified to work with numpy
            c1 = (x2-x1)*(yp-y1)-(y2-y1)*(xp-x1)
            c2 = (x3-x2)*(yp-y2)-(y3-y2)*(xp-x2)
            c3 = (x1-x3)*(yp-y3)-(y1-y3)*(xp-x3)
            intri = ((c1<=0) & (c2<=0) & (c3<=0)) | ((c1>=0) & (c2>=0) & (c3>=0))
            return intri

        x = np.array(x, np.float32)
        y = np.array(y, np.float32)
        #print 'inx',x
        #print 'iny',y

        predx = np.ones((len(x),)) * float('nan')
        predy = np.ones((len(y),)) * float('nan')
        for tri,trans in self.tris:
            #print tri
            (x1,y1),(x2,y2),(x3,y3) = tri
            intri = point_in_tri(x1,y1,x2,y2,x3,y3, x,y)
            trix = x[intri]
            triy = y[intri]
            if len(trix):
                #print len(trix)
                predtrix,predtriy = trans.predict(trix, triy)
                predx[intri] = predtrix
                predy[intri] = predtriy

        return predx, predy




    
