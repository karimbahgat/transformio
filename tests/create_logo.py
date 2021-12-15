
import transformio as tio
from PIL import Image
import numpy as np

im = Image.open('data/land_shallow_topo_2048.png')

bounds = [-180,90,180,-90] # left,upper,right,bottom
img2geo = tio.imwarp.fitbounds(im.size[0], im.size[1], bounds)

fromcrs = '+proj=longlat +datum=WGS84 +no_defs' #'epsg:4326'
tocrs = '+proj=ortho +lat_0=0 +lon_0=30 +x_0=30 +y_0=0' #'esri:54009'
geo2proj = tio.transforms.MapProjection(fromcrs, tocrs)

warped,_ = tio.imwarp.warp(im, [img2geo,geo2proj])

# prep clouds
clouds = Image.open('data/cloud_combined_2048.jpg')
clouds_alpha = clouds.convert('L')
clouds_overlay = Image.new('RGBA', im.size, (255,255,255,0))
clouds_overlay.paste((255,255,255,255), mask=clouds_alpha)
clouds_warped,_ = tio.imwarp.warp(clouds_overlay, [img2geo, geo2proj])

# final image
final = warped
#final = Image.new('RGB', warped.size, (0,0,0))
#final.paste(warped, mask=warped)
final.paste(clouds_warped, mask=clouds_warped)
final.save('logo-globe.png')
