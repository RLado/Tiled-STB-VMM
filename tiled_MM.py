import numpy as np
from PIL import Image
import cv2
import argparse
from utils import pad_img
import run
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


def vid2frames(vid_path, out_path = '.', crop = None): #Frame extractor function
    """
    Extracts frames from a video and saves them on a user designated directory.

    Parameters:
        vid_path (str): Path to the source video
        out_path (str): Path to output the extracted frames (the directory will 
            be created if it does not exist). Default = .
        crop (tuple): Crop region defined by a tuple containing a top left 
            coordinate and Width + Height, e.g. ((0,0),(100,100)). Default = None

    Returns:
        tuple: True if sucessful; Framerate; Frame count

    """

    vidObj = cv2.VideoCapture(vid_path)
    fps = vidObj.get(cv2.CAP_PROP_FPS) # Get framerate

    count = 0
    success = 1
    
    # Check if output path exisist, if not create directory
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    frames = []
    while True:
        success, image = vidObj.read()
        if success:
            # Saves the frames with frame-count
            if crop == None:
                cv2.imwrite(os.path.join(out_path,'frame_%06d.png' % count), image)
            else:
                cv2.imwrite(os.path.join(out_path,'frame_%06d.png' % count), image[crop[0][1]:crop[0][1]+crop[1][1],crop[0][0]:crop[0][0]+crop[1][0]])
            frames.append(os.path.join(out_path,'frame_%06d.png' % count))
            count += 1
        else:
            break
    
    return True, fps, count-1, frames


def tile(img, tile_size = 128, overlap = 30):
    # Load Image
    frame = Image.open(img).convert('RGB')

    # Calculate tiling stride
    stride = tile_size - overlap
    # - width
    nx_tiles=len(range(tile_size, frame.size[0]+tile_size, stride))
    x_pad = 0
    while (frame.size[0]+x_pad)%stride != 0:
        x_pad += 2

    # - height
    ny_tiles=len(range(tile_size, frame.size[1]+tile_size, stride))
    y_pad = 0
    while (frame.size[1]+y_pad)%stride != 0:
        y_pad += 2

    # Pad image
    frame = pad_img.pad_img(frame, frame.size[0]+x_pad+overlap, frame.size[1]+y_pad+overlap)

    # Convert to OpenCV format
    frame = np.array(frame) 
    # Convert RGB to BGR 
    frame = frame[:, :, ::-1].copy()

    # Break into tiles
    tiles = []
    for i in range(tile_size, frame.shape[1]+1, stride): # X axis
        for j in range(tile_size, frame.shape[0]+1, stride): #Y axis
            tiles.append(frame[j-tile_size:j, i-tile_size:i])
            assert(tiles[-1].shape[0] == tile_size)
            assert(tiles[-1].shape[1] == tile_size)
    
    return tiles, frame.shape


def stitch(tiles, frame_shape, stride = 98):
    # Read tile_size
    tile_size = tiles[0].shape[0]

    # Generate placeholder black frame
    s = np.zeros(frame_shape, dtype='uint8')

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
        if len(range(tile_size, frame_shape[0]+1, stride))%2 != 0:
            t -= 1
        t += len(range(tile_size, frame_shape[0]+1, stride))

    return s


if __name__ == '__main__':
    tile_size = 192
    overlap = 30
    stride = tile_size-overlap
    vid_path = './test_vid_short.mp4'
    temp_path = 'tiled_fragments'
    out_path = './tiled_result.avi'

    print('Extracting frames...')
    _, fps, _, frames = vid2frames(vid_path, out_path=temp_path)

    print('Splitting tiles...')
    tiles_files = []
    for i, f in enumerate(frames):
        tiles_files.append([])
        tiles, frame_shape = tile(f, tile_size, overlap)
        for j, t in enumerate(tiles):
            if not os.path.exists(os.path.join(temp_path,f'tile_{j}')):
                os.makedirs(os.path.join(temp_path,f'tile_{j}'))
            cv2.imwrite(os.path.join(temp_path,f'tile_{j}',f'fragment_{str(i).zfill(6)}.png'), t)
            tiles_files[-1].append(os.path.join(temp_path,f'tile_mag_{j}',f'STBVMM_static_{str(i).zfill(6)}.png'))
    #tiles_files = list(zip(*tiles_files))
    #print(tiles_files) #CHECK HERE AS WELL

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
    video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_shape[1], frame_shape[0]))
    for i in range(len(frames)-1):
        t = []
        for f in tiles_files[i]:
            t.append(cv2.imread(f))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        video.write(cv2.resize(stitch(t, frame_shape, stride = stride), (frame_shape[1], frame_shape[0])))
    video.release()
