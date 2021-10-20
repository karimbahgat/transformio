
import numpy as np
import PIL, PIL.Image
import math
import io
import urllib

from . import transforms


def imbounds(width, height, transform, gridsamples=20):
    # calc output bounds based on transforming source image pixel edges and diagonal distance, ala GDAL
    # TODO: alternatively based on internal grid or just all the pixels
    # see https://github.com/OSGeo/gdal/blob/60d8a9ca09c466225508cb82e30a64aefa899a41/gdal/alg/gdaltransformer.cpp#L135

    # NOTE: uses forward transform to calc output bounds, and backward transform for resampling
    # but for polynomial order >1 backward transform is reestimated on the points (inverse doesnt work)
    # and can be noticably different than the forward transform, thus miscalculating the bounds
    # TODO: maybe need a fix somehow...

    imw,imh = width, height
    pixels = []
    
    if gridsamples:
        # get sample pixels at intervals
        for row in range(0, imh+1, imh//gridsamples):
            for col in range(0, imw+1, imw//gridsamples):
                pixels.append((col,row))
        
        # ensure we get top and bottom edges
        for row in [0, imh+1]: # +1 to incl bottom of last row
            for col in range(0, imw+1, imw//gridsamples): # +1 to incl right of last column, ca 20 points along
                pixels.append((col,row))

        # ensure we get left and right edges
        for col in [0, imw+1]: # +1 to incl right of last column
            for row in range(0, imh+1, imh//gridsamples): # +1 to incl bottom of last row, ca 20 points along
                pixels.append((col,row))

    else:
        for row in range(0, imh+1):
            for col in range(0, imw+1):
                pixels.append((col,row))

    print(len(pixels))
    cols,rows = zip(*pixels)

    # transform and get bounds
    predx,predy = transform.predict(cols, rows)
    predx = predx[~np.isnan(predx)]
    predy = predy[~np.isnan(predy)]
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

def warp(im, transform, invtransform, resample='nearest', maxdim=None, crs=None):
    # check if im is url
    if isinstance(im, str) and im.startswith('http'):
        print('getting image from url')
        url = im
        fobj = io.BytesIO(urllib.request.urlopen(url).read())
        im = PIL.Image.open(fobj)

    # ensure correct im mode
    if not im.mode == 'RGBA':
        im = im.convert('RGBA')



    # create forward and backward chains, to easily add additional transforms
    transform = transforms.Chain(transforms=[transform]) # forward
    invtransform = transforms.Chain(transforms=[invtransform]) # backward



    # if downsampling original image
    if maxdim:
        longest = max(im.size)
        ratio = maxdim  / float(longest)
        if ratio < 1:
            # img is larger than maxdim
            print('downsizing')
            # resize
            nw,nh = int(im.size[0]*ratio), int(im.size[1]*ratio)
            im = im.resize((nw,nh), PIL.Image.ANTIALIAS)
            print(im)
            # chain resize transform with existing transform
            small2big = transforms.Polynomial(order=1,
                                                     A=[[1/ratio,0,0],
                                                        [0,1/ratio,0],
                                                        [0,0,1]],
                                                     )
            transform.transforms.insert(0, small2big)
            big2small = transforms.Polynomial(order=1,
                                                     A=[[ratio,0,0],
                                                        [0,ratio,0],
                                                        [0,0,1]],
                                                     )
            invtransform.transforms.append(big2small)



    # add in reproj EPSG 4326 -> EPSG X
    if crs:
        # forward
        fromcrs = '+init=epsg:4326' # this is the immediate coordsys of the transform estimation
        tocrs = '+init=epsg:' + crs
        crstrans = transforms.Projection(fromcrs=fromcrs, tocrs=tocrs)
        transform.transforms.append(crstrans)
        # backward
        fromcrs,tocrs = tocrs,fromcrs # switch from-to
        crstrans = transforms.Projection(fromcrs=fromcrs, tocrs=tocrs)
        invtransform.transforms.insert(0, crstrans)



    # get output bounds
    print(transform,invtransform)
    print('calculating coordinate bounds')
    imw,imh = im.size
    xmin,ymin,xmax,ymax = imbounds(imw, imh, transform)
    print(xmin,ymin,xmax,ymax)

    

    # calc diagonal dist and output dims
    dx,dy = xmax-xmin, ymax-ymin
    diag = math.hypot(dx, dy)
    xyscale = diag / float(math.hypot(imw, imh))
    w,h = int(dx / xyscale), int(dy / xyscale)

##    downsize = 10
##    w = int(w/float(downsize))
##    h = int(h/float(downsize))
##    xscale = dx / float(w)
##    yscale = dy / float(h)
    
    # set affine
    xoff,yoff = xmin,ymin
    xscale = yscale = xyscale 
    if True: #predy[0] > predy[-1]:    # WARNING: HACKY ASSUMES FLIPPED Y AXIS FOR NOW...
        yoff = ymax
        yscale *= -1
    affine = [xscale,0,xoff, 0,yscale,yoff]

    # resampling
    if resample == 'nearest':
    
        print('backwards mapping and resampling')
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
        cols = np.linspace(0, w-1, w)
        rows = np.linspace(0, h-1, h)
        cols,rows = np.meshgrid(cols, rows)
        cols,rows = cols.flatten(), rows.flatten()
        xs = xoff + (cols * xscale)
        ys = yoff + (rows * yscale)
        backpredx,backpredy = invtransform.predict(xs, ys)
        backpred = np.column_stack((backpredx, backpredy))
        backpred = backpred.reshape((h,w,2))
        
        print('writing to output')
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
        inarr = np.array(im)
        outarr = np.zeros((h, w, 4), dtype=np.uint8)
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
        outarr[mask] = inarr[backpred_rows[mask], backpred_cols[mask], :]

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

    out = PIL.Image.fromarray(outarr)
    return out, affine





    
