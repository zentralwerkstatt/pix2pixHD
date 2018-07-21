import random
import urllib.request
from subprocess import call
from PIL import Image
import numpy as np
import struct

z = 18
w = 600
h = 640
m = 2500

# Google Static Maps API with styled map: https://mapstyle.withgoogle.com/
key = "" # See untracked key.txt for key
sat = "https://maps.googleapis.com/maps/api/staticmap?key=" + key + "&center={lon:.6f},{lat:.6f}&zoom={zoom}&size={width}x{height}&maptype=satellite&key=AIzaSyA0WsMIhjNNYIrpvtWcUSE3broz7WXwq1Q&scale=2"
map="https://maps.googleapis.com/maps/api/staticmap?key=" + key + "&center={lon:.6f},{lat:.6f}&zoom={zoom}&format=png&maptype=roadmap&style=element:labels%7Cvisibility:off&style=feature:administrative%7Cvisibility:off&style=feature:administrative%7Celement:geometry%7Cvisibility:off&style=feature:landscape%7Cvisibility:off&style=feature:landscape%7Celement:geometry%7Cvisibility:off&style=feature:landscape%7Celement:labels%7Cvisibility:off&style=feature:landscape.man_made%7Cvisibility:off&style=feature:landscape.man_made%7Celement:geometry%7Cvisibility:simplified&style=feature:landscape.man_made%7Celement:labels%7Cvisibility:off&style=feature:landscape.natural%7Cvisibility:off&style=feature:landscape.natural%7Celement:geometry%7Ccolor:0xffffff%7Cvisibility:simplified&style=feature:landscape.natural.landcover%7Ccolor:0xffffff%7Cvisibility:simplified&style=feature:landscape.natural.terrain%7Cvisibility:off&style=feature:poi%7Cvisibility:off&style=feature:road%7Ccolor:0x000000%7Cvisibility:on&style=feature:road%7Celement:geometry%7Ccolor:0x000000%7Cvisibility:simplified&style=feature:road%7Celement:labels%7Cvisibility:off&style=feature:transit%7Cvisibility:off&style=feature:water%7Ccolor:0x0000ff%7Cvisibility:off&style=feature:water%7Celement:geometry%7Ccolor:0x0000ff%7Cvisibility:simplified&style=feature:water%7Celement:labels%7Cvisibility:off&size={width}x{height}&scale=2"

# Generate quantization and remapping map

hex_colors = {'DEDEDE':'FFFFFF', 'F2F2F2':'FF0000', '000000':'000000', '0000FF':'0000FF', 'FBEDD6':'FF0000', 'FFFFFF':'FFFFFF'}
palette = np.zeros((10,60,3), dtype=np.uint8)

i=0
for hex_color, hex_replacement in hex_colors.items():
    rgb_color = struct.unpack('BBB', bytes.fromhex(hex_color))
    palette[:,i:i+10,0] = rgb_color[0]
    palette[:,i:i+10,1] = rgb_color[1]
    palette[:,i:i+10,2] = rgb_color[2]
    i+=10

im = Image.fromarray(palette)
im.save('hd_palette.png')

n = 0

while n<=m:
  
    # ~Kyoto
    # random_lon = random.uniform(34.8, 35.0)
    # random_lat = random.uniform(135.6, 135.8)
    
    # ~Stuttgart
    random_lon = random.uniform(48.74, 48.78)
    random_lat = random.uniform(9.19, 9.23)
    
    url_sat = sat.format(zoom=z, lon=random_lon, lat=random_lat, width=w, height=h)
    url_map = map.format(zoom=z, lon=random_lon, lat=random_lat, width=w, height=h)

    file_sat = 'min_map_720/A/{0}.png'.format(n)
    urllib.request.urlretrieve(url_sat, file_sat)    
    
    # Increase brightness/saturation and crop
    call('mogrify -modulate 150,150 -crop 1200x1200+0+0 +repage -flatten ' + file_sat, shell=True)
    
    # Resize
    call('mogrify -resize 1280x1280 -crop 1280x720+0+0 ' + file_sat, shell=True)
    
    file_map = 'min_map_720/B/{0}.png'.format(n)
    urllib.request.urlretrieve(url_map, file_map)

    # Crop and quantize
    call('mogrify -crop 1200x1200+0+0 +repage -dither None -remap hd_palette.png ' + file_map, shell=True)

    # Remap
    for hex_color, hex_replacement in hex_colors.items():
        call('mogrify -fill "#' + hex_replacement + '" -opaque "#'+ hex_color +'" ' + file_map, shell=True)
        
    # Resize
    call('mogrify -resize 1280x1280 -crop 1280x720+0+0 ' + file_map, shell=True)

    n+=1