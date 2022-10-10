# Tiled STB-VMM

Sometimes [STB-VMM](https://github.com/RLado/STB-VMM 'repo') uses too much memory and becomes unusable in some setups, especially when magnifying large resolution videos. To mitigate this problem, this code breaks the sequence into multiple slightly overlapping tiles that are then magnifyed individually and stitched together at the end, yielding acceptable results using much less memory.

### [STB-VMM](https://github.com/RLado/STB-VMM 'repo') Demo

https://user-images.githubusercontent.com/25719985/194240973-8d93968f-283b-4802-aacb-5e32175e16f3.mp4

## Setup
### Clone submodules
```bash
git submodule update --init --recursive
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### [Download](https://github.com/RLado/STB-VMM/releases/tag/v1.0.0) trained STB-VMM model
For this example we place the checkpoint in *./STB-VMM/ckpt/ckpt_e49.pth.tar*. You may place this file anywhere you like.

## How to run
To magnify x20 a *sample_video.mp4* with a tile size of 512px use:

```bash
python magnify.py -i sample_video.mp4 -c STB-VMM/ckpt/ckpt_e49.pth.tar -m 20 -t 512 
```

For more information regarding options and parameters read the output of ```magnify.py -h```:

```
Tiled STB-VMM: Break large videos into tiles, magnify those tiles and stitch'em together. Makes large videos processable with
low amounts of RAM

options:
  -h, --help            show this help message and exit
  -i PATH, --video_path PATH
                        path to video input frames
  --temp PATH           path to save temporal data (deleted on exit) (default: /dev/shm/temp_STB-VMM)
  -c PATH, --load_ckpt PATH
                        path to load checkpoint
  -o PATH, --output PATH
                        path to save generated frames (default: demo.avi)
  -m N, --mag N         magnification factor (default: 20.0)
  --mode {static,dynamic}
                        magnification mode (static, dynamic)
  -t T, --tile_size T   size of the tiles to be processed. The bigger the tile the faster magnification runs, as long as the
                        tile fits in VRAM
  --overlap O           tile edge overlap in pixels (default: 30)
  -j N, --workers N     number of data loading workers (default: 16)
  -b N, --batch_size N  batch size (default: 1)
  -p N, --print_freq N  print frequency (default: 100)
  --device {auto,cpu,cuda}
                        select device [auto/cpu/cuda] (default: auto)

```