import numpy as np
from PIL import Image
import cv2
import argparse
from utils import pad_img
import run
import video2frames
import os

class STB_args:
    def __init__(
            self,
            amp, 
            video_path, 
            save_dir,
            load_ckpt,
            num_data,
            mode = 'static',
            device = 'auto',
            workers = 4, 
            batch_size = 1,
            print_freq = 1000
        ):
        self.workers = workers
        self.batch_size = batch_size
        self.load_ckpt = load_ckpt
        self.save_dir = save_dir
        self.device = device
        self.amp = amp
        self.mode = mode
        self.video_path = video_path
        self.num_data = num_data
        self.print_freq = print_freq

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
    tile_size = 512
    overlap = 30
    stride = tile_size-overlap
    vid_path = './test_vid_short.mp4'
    temp_path = './tiled_fragments'
    out_path = './tiled_result'

    print('Extracting frames...')
    video2frames.vid2frames(vid_path, out_path=temp_path)

    # List all frames of the input video
    frames = [f for f in os.listdir(temp_path) if f.endswith('.png')]

    print('Splitting tiles...')
    for i, f in enumerate(frames):
        print(i)
        tiles, frame_shape = tile(os.path.join(temp_path, f), tile_size, overlap)
        for j, t in enumerate(tiles):
            if not os.path.exists(os.path.join(temp_path,f'tile_{j}')):
                os.makedirs(os.path.join(temp_path,f'tile_{j}'))
            cv2.imwrite(os.path.join(temp_path,f'tile_{j}',f'fragment_{str(i).zfill(6)}.png'), t)

    print('Computing magnification...')
    for j in range(len(tiles)):
        stb_args = STB_args(
            amp = 20,
            video_path = os.path.join(temp_path,f'tile_{j}')+'/fragment',
            save_dir = os.path.join(temp_path,f'tile_mag_{j}'),
            load_ckpt = './ckpt/ckpt_e49.pth.tar',
            num_data = len(frames)-2,
            mode = 'static',
            device = 'auto',
            workers = 4, 
            batch_size = 1,
            print_freq = 10
        )
        run.main(stb_args)

    print('Stitching tiles...')
    tiles_files = []
    for j in range(6):#(len(tiles)):
        tiles_files.append([os.path.join(temp_path,f'tile_mag_{j}',f) for f in os.listdir(os.path.join(temp_path,f'tile_mag_{j}')) if f.endswith('.png')])
    tiles_files = list(zip(*tiles_files))
    for i in range(119):#(len(frames)-1):
        t = []
        for f in tiles_files[0]:
            t.append(cv2.imread(f))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(os.path.join(out_path, f'result_{str(i).zfill(6)}.png'), stitch(t, frame_shape, stride = stride))