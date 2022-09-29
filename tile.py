import numpy as np
from PIL import Image
import cv2
import argparse
from STB_VMM.utils import pad_img


def tile(img, tile_size = 128, overlap = 30):
    # Load Image
    frame = Image.open(img).convert('RGB')

    # Calculate tiling stride
    stride = tile_size - overlap
    # - width
    nx = 0 # number of tiles - 1 on X
    x_pad = frame.size[1]
    while x_pad > tile_size:
        nx += 1
        x_pad = frame.size[0] - (tile_size+nx*(tile_size-overlap))
    # - height
    ny = 0 # number of tiles - 1 on Y
    y_pad = frame.size[1]
    while y_pad > tile_size:
        ny += 1
        y_pad = frame.size[1] - (tile_size+ny*(tile_size-overlap))

    # Pad image
    frame = pad_img.pad_img(frame, x_pad, y_pad)

    # Convert to OpenCV format
    frame = np.array(frame) 
    # Convert RGB to BGR 
    frame = frame[:, :, ::-1].copy()
    print(frame.shape)

    # Break into tiles
    tiles = []
    for i in range(tile_size, frame.shape[1]+1, stride): # X axis
        for j in range(tile_size, frame.shape[0]+1, stride): #Y axis
            tiles.append(frame[j-tile_size:j, i-tile_size:i])
    
    return tiles, frame.shape


def stitch(tiles, frame_shape, stride = 98):
    # Read tile_size
    tile_size = tiles[0].shape[0]

    # Generate placeholder black frame
    s = np.zeros(frame_shape)

    t = 0
    for i in range(tile_size, frame_shape[1]+1, stride): # X axis
        for j in range(tile_size, frame_shape[0]+1, stride): #Y axis
            s[j-tile_size:j, i-tile_size:i] = tiles[t]
            t += 1
    t = 0
    for i in range(tile_size, frame_shape[1]+1, stride*2): # X axis
        for j in range(tile_size, frame_shape[0]+1, stride*2): #Y axis
            s[j-tile_size:j, i-tile_size:i] = s[j-tile_size:j, i-tile_size:i]*.5 + tiles[t]*.5
            t += 2
        t += len(range(tile_size, frame_shape[0]+1, stride)) - 1

    return s


if __name__ == '__main__':
    tile_size = 128
    overlap = 30
    stride = tile_size-overlap

    tiles, frame_shape = tile('frame_test.png', tile_size, overlap)
    cv2.imwrite('result.png', stitch(tiles, frame_shape, stride = 98))