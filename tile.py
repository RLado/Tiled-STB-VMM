import numpy as np
from PIL import Image
import cv2
import argparse
import img_stitcher
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

if __name__ == '__main__':
    tile_size = 128
    overlap = 30
    stride = tile_size-overlap

    tiles, frame_shape = tile('frame_test.png', tile_size, overlap)
    s = np.zeros((tile_size, tile_size, 3))

    t = 0
    for i in range(0, frame_shape[1]-tile_size, stride): # X axis
        for j in range(0, frame_shape[0]-tile_size, stride): #Y axis
            s = img_stitcher.stitch(
                    img_stitcher.img(
                        s,
                        (0, 0),
                        (s.shape[1], s.shape[0])
                    ),
                    img_stitcher.img(
                        tiles[t],
                        (i, j),
                        (tile_size, tile_size)
                    ),
                )
            t += 1
            print(t)

    cv2.imwrite('result.png', s)


