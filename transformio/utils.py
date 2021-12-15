
from PIL import Image, ImageDraw

from . import vector
from . import imwarp

def draw_geojson(geoj, im=None, imbounds=None, fillcolor='blue', fillsize=10, outlinecolor=None, outlinewidth=1):
    # input geojson can be a single geojson or a list of geojsons
    geojs = geoj if isinstance(geoj, list) else [geoj]

    # create image bounds based on geojson bounds if not already given
    if imbounds is None:
        x1,y1,x2,y2 = vector.get_bbox(geoj) # TODO: SHOULD BE geojs
        w,h = x2-x1,y2-y1
        pad = 0.05 # 5 percent
        imbounds = x1-w*pad, y2+h*pad, x2+w*pad, y1-h*pad

    # create image if not already given
    if im is None:
        # make sure imbounds fits inside image aspect ratio
        aspect = h/w
        width = 1000.0
        height = width * aspect
        width,height = map(int, (width,height))

        # create image
        im = Image.new('RGBA', (width,height))
    
    # create transform
    width,height = im.size
    im2geo = imwarp.fitbounds(width, height, imbounds)
    geo2im = im2geo.inverse()

    # create the drawer
    drawer = ImageDraw.Draw(im)

    # define the drawing funcs
    def draw(geoj):
        typ = geoj['type']
        if "Point" in typ:
            halfwidth = halfheight = fillsize/2.0
            points = geoj['coordinates'] if "Multi" in typ else [geoj['coordinates']]
            for x,y in points:
                bbox = [x-halfwidth, y-halfheight, x+halfwidth, y+halfheight]
                drawer.ellipse(bbox, fill=fillcolor, outline=outlinecolor, width=outlinewidth)
        else:
            raise NotImplementedError()

    # loop the geojsons
    for geoj in geojs:
        # transform geojson coordinates to image coordinates
        geoj = vector.transform(geoj, geo2im)

        # draw
        draw(geoj)

    # return
    return im




