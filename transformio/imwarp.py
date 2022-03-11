
import numpy as np
import PIL, PIL.Image
import math
import io
import urllib

from . import transforms

def fitbounds(width, height, bounds):
    '''Fit and return a pixel-to-coordinate affine transformation 
    for an image whose corner coordinates are known. 
    
    Note: This is the inverse of imbounds(). 
    '''
    # define corner coordinates in pixel and geographic space
    imgcorners = [(0,0),(width,0),(width,height),(0,height)]
    bx1,by1,bx2,by2 = bounds
    if bounds[1] < bounds[3]:
        # flip y axis
        bx1,by1,bx2,by2 = bx2,by2,bx1,by1
    geocorners = [(bx1,by1),(bx2,by1),(bx2,by2),(bx1,by2)]
    x1,y1 = zip(*imgcorners)
    x2,y2 = zip(*geocorners)
    # based on these create the image to geographic transform
    trans = transforms.Polynomial(order=1)
    trans.fit(x1, y1, x2, y2)
    return trans

def imbounds(width, height, transform):
    '''Calculate the output bounds by transforming source image pixel 
    coordinates. 
    '''
    # NOTE: This is the same approach used by GDAL
    # TODO: alternatively based on internal grid or just all the pixels
    # see https://github.com/OSGeo/gdal/blob/60d8a9ca09c466225508cb82e30a64aefa899a41/gdal/alg/gdaltransformer.cpp#L135

    # NOTE: uses forward transform to calc output bounds, and backward transform for resampling
    # but for polynomial order >1 backward transform is reestimated on the points (inverse doesnt work)
    # and can be noticably different than the forward transform, thus miscalculating the bounds
    # TODO: maybe need a fix somehow...

    # get sample pixels at intervals
    imw,imh = width,height
    cols = np.linspace(0, imw, 100)
    rows = np.linspace(0, imh, 100)
    cols,rows = np.meshgrid(cols, rows)
    cols,rows = cols.flatten(), rows.flatten()

    # ensure we get every pixel along edges
    allcols = np.linspace(0, imw, imw+1)
    allrows = np.linspace(0, imh, imh+1)
    # top
    cols = np.append(cols, allcols)
    rows = np.append(rows, np.zeros(allcols.shape))
    # bottom
    cols = np.append(cols, allcols)
    rows = np.append(rows, np.zeros(allcols.shape)*imh)
    # left
    cols = np.append(cols, np.zeros(allrows.shape))
    rows = np.append(rows, allrows)
    # right
    cols = np.append(cols, np.zeros(allrows.shape)*imw)
    rows = np.append(rows, allrows)

    # transform and get bounds
    predx,predy = transform.predict(cols, rows)
    predx = predx[~np.isnan(predx)]
    predy = predy[~np.isnan(predy)]
    predx = predx[~np.isinf(predx)]
    predy = predy[~np.isinf(predy)]
    xmin,ymin,xmax,ymax = predx.min(), predy.min(), predx.max(), predy.max()

    # (TEMP GET WGS84 BOUNDS TOO)
##    transform.transforms.pop(-1)
##    predx,predy = transform.predict(cols, rows)
##    predx = predx[~np.isnan(predx)]
##    predy = predy[~np.isnan(predy)]
##    raise Exception(str((predx.min(), predy.min(), predx.max(), predy.max())))

    # TODO: maybe walk along output edges and backwards transform
    # to make sure covers max possible of source img
    # in case forward/backward transform is slightly mismatched
    # ...

    return xmin,ymin,xmax,ymax

def warp(im, transform, resample='nearest', size=None, bounds=None, maxdim=None, fromcrs=None, tocrs=None):
    # transform can be a single transform instance, a chain transform, or a list of transforms.
    # check if im is url
    # TODO: Probably drop maxdim arg... 
    if isinstance(im, str) and im.startswith('http'):
        #print('getting image from url')
        url = im
        fobj = io.BytesIO(urllib.request.urlopen(url).read())
        im = PIL.Image.open(fobj)



    # create transform chain, to easily add additional transforms
    if isinstance(transform, list):
        chain = transforms.Chain(transforms=transform) 
    elif isinstance(transform, transforms.Chain):
        chain = transform.copy()
    else:
        chain = transforms.Chain(transforms=[transform]) 



    # if downsampling original image
    if maxdim:
        longest = max(im.size)
        ratio = maxdim  / float(longest)
        if ratio < 1:
            # img is larger than maxdim
            #print('downsizing')
            # resize
            nw,nh = int(im.size[0]*ratio), int(im.size[1]*ratio)
            im = im.resize((nw,nh), PIL.Image.ANTIALIAS)
            #print(im)
            # chain resize transform with existing transform
            big2small = transforms.Polynomial(order=1,
                                                     A=[[1/ratio,0,0],
                                                        [0,1/ratio,0],
                                                        [0,0,1]],
                                                     )
            chain.transforms.insert(0, big2small)



    # add in reproj
    if fromcrs and tocrs:
        # forward
        crstrans = transforms.MapProjection(fromcrs=fromcrs, tocrs=tocrs)
        chain.add(crstrans)



    # get output bounds
    #print('calculating coordinate bounds')
    imw,imh = im.size
    if bounds:
        # user defined bounds
        xmin,ymin,xmax,ymax = bounds
    else:
        # calc bounds by forward projecting image pixels
        xmin,ymin,xmax,ymax = imbounds(imw, imh, chain)
    #print(xmin,ymin,xmax,ymax)

    

    # get output dims
    if size:
        # user-specified output dimensions
        dx,dy = xmax-xmin, ymax-ymin
        w,h = size
        xscale = dx / w
        yscale = dy / h
    else:
        # calc output size to approx match input size (based on diagonal)
        dx,dy = xmax-xmin, ymax-ymin
        diag = math.hypot(dx, dy)
        xscale = yscale = diag / float(math.hypot(imw, imh))
        w,h = int(dx / xscale), int(dy / yscale)

##    downsize = 10
##    w = int(w/float(downsize))
##    h = int(h/float(downsize))
##    xscale = dx / float(w)
##    yscale = dy / float(h)
    
    # set output affine
    xoff,yoff = xmin,ymin 
    if True: #predy[0] > predy[-1]:    # WARNING: HACKY ASSUMES FLIPPED Y AXIS FOR NOW...
        yoff = ymax
        yscale *= -1
    affine = [xscale,0,xoff, 0,yscale,yoff]

    # resampling
    if resample == 'nearest':

##        print 'experimental forward resampling'
##        # this shows where forward mapping would put each pixel, and defines the output bounds
##        # but the actual backward resampling for poly2/3 is often widely different (compare to see)
##        pixels = []
##        for row in range(imh):
##            for col in range(imw):
##                pixels.append((col,row))
##        cols,rows = zip(*pixels)
##        xs,ys = transform.predict(cols, rows)
##        _A = np.eye(3)
##        _A[0,:] = affine[:3]
##        _A[1,:] = affine[3:6]
##        _Ainv = np.linalg.inv(_A)
##        terms = np.array([xs, ys, np.ones(xs.shape)])
##        cols2,rows2 = _Ainv.dot(terms)[:2]
##        cols2,rows2 = np.floor(cols2).astype(int), np.floor(rows2).astype(int)
##        cols2,rows2 = np.clip(cols2, 0, w-1), np.clip(rows2, 0, h-1)
##        # write
##        outarr = np.zeros((h, w, 4), dtype=np.uint8)
##        inarr = np.array(im)
##        outarr[rows2,cols2,:] = inarr[rows,cols,:]
##
##        PIL.Image.fromarray(outarr).show()
    
        #print('backwards mapping and resampling')
##        coords = []
##        for row in range(h):
##            y = yoff + row*yscale
##            for col in range(w):
##                x = xoff + col*xscale
##                coords.append((x,y))
##        xs,ys = zip(*coords)
##        backpredx,backpredy = invtransform.predict(xs, ys)
##        backpred = np.column_stack((backpredx, backpredy))
##        backpred = backpred.reshape((h,w,2))
        # define output pixels
        cols = np.linspace(0, w-1, w)
        rows = np.linspace(0, h-1, h)
        cols,rows = np.meshgrid(cols, rows)
        cols,rows = cols.flatten(), rows.flatten()
        # convert to output coordinates
        xs = xoff + (cols * xscale)
        ys = yoff + (rows * yscale)
        # backwards predict output coords to original input pixels
        invchain = chain.inverse()
        #print('inverse chain',invchain)
        backpredx,backpredy = invchain.predict(xs, ys)
        #info = []
        #info.append(('target affine', affine))
        #info.append(('target bounds', bounds))
        #info.append(('backpred bounds', backpredx.min(), backpredy.min(), backpredx.max(), backpredy.max()))

        # cropping only the relevant parts of the image
        if bounds:
            xvalid, yvalid = ~np.isnan(backpredx), ~np.isnan(backpredy)
            backpred_xmin, backpred_ymin = backpredx[xvalid].min(), backpredy[yvalid].min()
            backpred_xmax, backpred_ymax = backpredx[xvalid].max(), backpredy[yvalid].max()
            cropbox = max(backpred_xmin, 0), max(backpred_ymin, 0), min(backpred_xmax, imw-1), min(backpred_ymax, imh-1)
            cropbox = tuple([int(round(v)) for v in cropbox])
            #info.append(('cropping to', cropbox))
            im = im.crop(cropbox)
            # update input image size and offset the backpred coords
            imw,imh = im.size
            #info.append(('imdims',imw,imh))
            if backpred_xmin > 0:
                backpredx -= backpred_xmin
            if backpred_ymin > 0:
                backpredy -= backpred_ymin
            #info.append(('backpred bounds', backpredx.min(), backpredy.min(), backpredx.max(), backpredy.max()))
            #info.append(im)
        
        #print('writing to output')
        # slow, can prob optimize even more by using direct numpy indexing
        # 4 bands, fourth is the alpha, invisible for pixels that were not sampled
        # currently assumes input image is RGBA only... 
##        outarr = np.zeros((h, w, 4), dtype=np.uint8)
##        imload = im.load()
##        for row in range(h):
##            for col in range(w):
##                origcol,origrow = backpred[row,col]
##                if math.isnan(origcol) or math.isnan(origrow):
##                    continue
##                origcol,origrow = int(math.floor(origcol)), int(math.floor(origrow))
##                if 0 <= origcol < imw and 0 <= origrow < imh:
##                    rgba = list(imload[origcol,origrow])
##                    #rgba[-1] = 255 # fully visible
##                    outarr[row,col,:] = rgba

        # faster numpy version
        if not im.mode.startswith('RGB'):
            # special color formats must be converted to rgb
            if im.mode.endswith('A'):
                # only convert to alpha if necessary
                im = im.convert('RGBA')
            else:
                im = im.convert('RGB')
        inarr = np.array(im)
        outarr = np.zeros((h, w, 4), dtype=np.uint8)
        backpred = np.column_stack((backpredx, backpredy))
        backpred = backpred.reshape((h,w,2))
        backpred_cols = backpred[:,:,0]
        backpred_rows = backpred[:,:,1]
        # valid
        backpred_valid = ~(np.isnan(backpred_cols) | np.isnan(backpred_rows))
        # nearest pixel rounding
        backpred_cols = np.around(backpred_cols, 0).astype(int)
        backpred_rows = np.around(backpred_rows, 0).astype(int)
        # define image bounds
        backpred_inbounds = (backpred_cols >= 0) & (backpred_cols < imw) & (backpred_rows >= 0) & (backpred_rows < imh)
        # do the sampling
        mask = (backpred_valid & backpred_inbounds)
        #info.append(('final backpred bounds', backpred_cols[mask].min(), backpred_rows[mask].min(), backpred_cols[mask].max(), backpred_rows[mask].max()))
        if len(im.mode) == 4:
            # inarr and outarr both have rgba
            outarr[mask] = inarr[backpred_rows[mask], backpred_cols[mask], :]
        elif len(im.mode) == 3:
            # inarr only has rgb
            outarr[mask,:3] = inarr[backpred_rows[mask], backpred_cols[mask], :] # set rgb only
            outarr[mask,-1] = 255 # set alpha to visible
        else:
            raise Exception('Unexpected image mode: {}'.format(im.mode))

        # memory friendly numpy version
        # NOT FINISHED YET... 
        '''
        # figure out roughly how many chunks of outputs 
        # in order to get 256x256 chunks in input
        backpred_xmin, backpred_ymin = backpredx.min(), backpredy.min()
        backpred_xmax, backpred_ymax = backpredx.max(), backpredy.max()
        backpred_width = backpred_xmax-backpred_xmin
        backpred_height = backpred_ymax-backpred_ymin
        xsteps = backpred_width / 256 # how many horizontal 256 blocks fits in input
        ysteps = backpred_height / 256 # how many vertical 256 blocks fits in input
        xstepsize,ystepsize = int(w // xsteps), int(h // ysteps) # get output tilesize by dividing by that many blocks
        print('output stepsize', xstepsize, ystepsize)

        # iterate each chunk of output (roughly equivalent of 256x256 in input)
        outarr = np.zeros((h, w, 4), dtype=np.uint8)
        backpred = np.column_stack((backpredx, backpredy))
        backpred = backpred.reshape((h,w,2))
        for tiley in range(0, h, ystepsize):
            tiley2 = min(tiley+ystepsize-1, h-1)
            for tilex in range(0, w, xstepsize):
                tilex2 = min(tilex+xstepsize-1, w-1)
                tilew,tileh = tilex2-tilex, tiley2-tiley
                print('tilebox',tilex,tilex2,tiley,tiley2)
                print('tilesize',tilew,tileh)
                # get predicted input coords for current output tile
                _backpred = backpred[tiley:tiley2, tilex:tilex2]
                backpred_cols = _backpred[:,:,0]
                backpred_rows = _backpred[:,:,1]
                # valid
                backpred_valid = ~(np.isnan(backpred_cols) | np.isnan(backpred_rows))
                # nearest pixel rounding
                backpred_cols = np.around(backpred_cols, 0).astype(int)
                backpred_rows = np.around(backpred_rows, 0).astype(int)
                # define image bounds
                backpred_inbounds = (backpred_cols >= 0) & (backpred_cols < w) & (backpred_rows >= 0) & (backpred_rows < h)
                # crop the image
                backpred_colmin, backpred_rowmin = backpred_cols.min(), backpred_rows.min()
                backpred_colmax, backpred_rowmax = backpred_cols.max(), backpred_rows.max()
                cropbox = backpred_colmin, backpred_rowmin, backpred_colmax, backpred_rowmax
                print('cropbox',cropbox)
                im = im.crop(cropbox)
                print('cropped',im)
                # offset so indices point to chunk indices
                backpred_cols -= backpred_colmin
                backpred_rows -= backpred_rowmin
                # convert to rgba
                if not im.mode == 'RGBA':
                    im = im.convert('RGBA')
                # do the sampling
                inarr = np.array(im)
                mask = (backpred_valid & backpred_inbounds)
                print('mask',mask.shape)
                print('outarr',outarr[tiley:tiley2, tilex:tilex2].shape)
                print('inarr',inarr.shape)
                #info.append(('final backpred bounds', backpred_cols[mask].min(), backpred_rows[mask].min(), backpred_cols[mask].max(), backpred_rows[mask].max()))
                outarr[tiley:tiley2, tilex:tilex2][mask] = inarr[backpred_rows[mask], backpred_cols[mask], :]
        '''

    else:
        raise ValueError('Unknown resample arg: {}'.format(resample))

    # calc warped target bounds?
##    w,h = img.size
##    A = [result['affine'][:3],
##         result['affine'][3:6],
##         [0,0,1]]
##    forward = mapfit.transforms.from_json({'type':'Polynomial', 'params':{}, 'data':{'A':A}})
##    x,y = forward.predict([0], [0])
##    x1,y1 = float(x[0]), float(y[0])
##    x,y = forward.predict([w], [h])
##    x2,y2 = float(x[0]), float(y[0])
##    result['bbox'] = [x1,y1,x2,y2]

    #print('\n'.join([str(e) for e in info]))

    out = PIL.Image.fromarray(outarr)
    return out, affine





    
