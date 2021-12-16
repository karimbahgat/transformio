
from PIL import Image, ImageDraw

from . import vector
from . import imwarp

def draw_geojson(geoj, im=None, imbounds=None, fillcolor='blue', fillsize=10, outlinecolor=None, outlinewidth=1):
    # create image bounds based on geojson bounds if not already given
    if imbounds is None:
        x1,y1,x2,y2 = vector.get_bbox(geoj)
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
    def draw_geom(geoj):
        typ = geoj['type']
        if "Point" in typ:
            halfwidth = halfheight = fillsize/2.0
            points = geoj['coordinates'] if "Multi" in typ else [geoj['coordinates']]
            for x,y in points:
                bbox = [x-halfwidth, y-halfheight, x+halfwidth, y+halfheight]
                drawer.ellipse(bbox, fill=fillcolor, outline=outlinecolor, width=outlinewidth)
        elif 'Polygon' in typ:
            polys = geoj['coordinates'] if "Multi" in typ else [geoj['coordinates']]
            for poly in polys:
                # exterior
                ext = poly[0]
                drawer.polygon(ext, fill=fillcolor, outline=False)
                # have to draw separate outline bc polygon doesn't allow customizing the outline width
                drawer.line(ext, fill=outlinecolor, width=outlinewidth) 
        else:
            raise NotImplementedError()

    # get list of geometries from geojson
    geotype = geoj['type']
    if geotype == 'FeatureCollection':
        geoms = [feat['geometry'] for feat in geoj['features']]
    elif geotype == 'GeometryCollection':
        geoms = geoj['geometries']
    elif geotype == 'Feature':
        geoms = [geoj['geometry']]
    else:
        geoms = [geoj]

    # loop the geometries
    for geom in geoms:
        # transform geometry coordinates to image coordinates
        geom = vector.transform(geom, geo2im)

        # draw
        draw_geom(geom)

    # return
    return im, imbounds




