
import numpy as np

def iter_points(geoj):
    """Yields every point in the geometry as a flat generator,
    useful for quick inspecting of dimensions so don't have to
    worry about nested coordinates and geometry types. For polygons
    this includes the coordinates of both exterior and holes."""

    def iter_geometry(geoj):
        geotype = geoj["type"]
        coords = geoj["coordinates"]
        
        if geotype == "Point":
            yield geoj["coordinates"]
        elif geotype in ("MultiPoint","LineString"):
            for p in coords:
                yield p
        elif geotype == "MultiLineString":
            for line in coords:
                for p in line:
                    yield p
        elif geotype == "Polygon":
            for ext_or_hole in coords:
                for p in ext_or_hole:
                    yield p
        elif geotype == "MultiPolygon":
            for poly in coords:
                for ext_or_hole in poly:
                    for p in ext_or_hole:
                        yield p

    geotype = geoj["type"]
    if geotype == 'FeatureCollection':
        for feat in geoj['features']:
            for p in iter_geometry(feat['geometry']):
                yield p
    
    elif geotype == 'GeometryCollection':
        for geom in geoj['geometries']:
            for p in iter_geometry(geom):
                yield p

    elif geotype == 'Feature':
        for p in iter_geometry(geoj['geometry']):
            yield p
    
    else:
        for p in iter_geometry(geoj):
            yield p
    

def get_bbox(geoj):
    xs,ys = zip(*iter_points(geoj))
    bbox = [min(xs),min(ys),max(xs),max(ys)]
    return bbox

def transform(geoj, transform):
    """
    Returns a transformed copy of the input feature geojson, using the supplied Transform instance. 
    For points, the function is applied to a list containing a single coordinate, separately for each multipart. 
    For linestrings, the function is applied to each multipart. 
    For polygons, the function is applied to the exterior, and to each hole if any. 
    """
    if not geoj:
        return None

    def transform_geometry(geoj):
        geotype = geoj["type"]
        coords = geoj["coordinates"]
        out = {'type':geotype, 'coordinates':[]}

        def wrapfunc(coords):
            # transform coords using the transform
            x,y = zip(*coords)
            xpred,ypred = transform.predict(x, y)
            # only keep points that are not inf or nan
            invalid = np.isnan(xpred) | np.isnan(ypred) | np.isinf(xpred) | np.isinf(ypred)
            #raise Exception(str(invalid))
            xpred,ypred = xpred[~invalid],ypred[~invalid]
            coords = list(zip(xpred,ypred))
            return coords
        
        if geotype == "Point":
            coords = wrapfunc([coords])
            if coords:
                out["coordinates"] = coords[0]
            else:
                pass
        elif geotype in ("MultiPoint","LineString"):
            out["coordinates"] = wrapfunc(coords)
            if not out["coordinates"]:
                pass
        elif geotype == "MultiLineString":
            coords = [wrapfunc(line)
                        for line in coords]
            out["coordinates"] = [line for line in coords if line]
            if not any(out["coordinates"]):
                pass
        elif geotype == "Polygon":
            coords = [wrapfunc(ext_or_hole)
                        for ext_or_hole in coords]
            out["coordinates"] = [ext_or_hole for ext_or_hole in coords if ext_or_hole]
            if not any(out["coordinates"]):
                pass
        elif geotype == "MultiPolygon":
            coords = [[wrapfunc(ext_or_hole)
                        for ext_or_hole in poly]
                        for poly in coords]
            coords = [[ext_or_hole
                        for ext_or_hole in poly if ext_or_hole]
                        for poly in coords]
            out["coordinates"] = [poly
                                    for poly in coords if poly]
            if not any(out["coordinates"]):
                pass
        
        return out

    geotype = geoj["type"]
    if geotype == 'FeatureCollection':
        out = {'type':'FeatureCollection', 'features':[]}
        for feat in geoj['features']:
            props = feat['properties'].copy()
            newgeom = transform_geometry(feat['geometry'])
            newfeat = {'type':'Feature', 'properties':props, 'geometry':newgeom}
            out['features'].append(newfeat)

    elif geotype == 'GeometryCollection':
        out = {'type':'GeometryCollection', 'geometries':[]}
        for geom in geoj['geometries']:
            newgeom = transform_geometry(geom)
            out['geometries'].append(newgeom)

    elif geotype == 'Feature':
        out = {'type':'Feature'}
        out['properties'] = geoj['properties'].copy()
        out['geometry'] = transform_geometry(geoj['geometry'])

    else:
        # single geometry
        out = transform_geometry(geoj)

    return out



