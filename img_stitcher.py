import numpy as np
import cv2
import argparse


class img:
    def __init__(self, data: np.ndarray, coord: tuple, dim: tuple):
        self.data: np.ndarray = data
        # This is inverted because OpenCV expects (y, x)
        self.coord: tuple = (coord[1], coord[0])
        # This is inverted because OpenCV expects (y, x)
        self.dim: tuple = (dim[1], dim[0])
        self.loc2glob: dict = {}
        self.build_local_mapings()

    def build_local_mapings(self):
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.loc2glob[(i, j)] = (i+self.coord[0], j+self.coord[1])


def average(a, b):  # merge function option
    c = []
    assert len(a) == len(b)
    for i in range(len(a)):
        c.append(a[i]*.5 + b[i]*.5)
    return np.array(c)


def stitch(imgA, imgB, mfun=average, out_path=None) -> np.ndarray:
    # Define output dimension
    max_c_Y = [imgA.coord[0] + imgA.dim[0],
               imgB.coord[0] + imgB.dim[0]]  # max coords
    max_c_X = [imgA.coord[1] + imgA.dim[1],
               imgB.coord[1] + imgB.dim[1]]  # max coords
    s = np.zeros((max(max_c_Y), max(max_c_X), 3))  # 3 channel images only

    # Invert loc2glob dictionary
    imgA_glob2loc = {v: k for k, v in imgA.loc2glob.items()}
    imgB_glob2loc = {v: k for k, v in imgB.loc2glob.items()}

    # Merge dictionary and write s
    for k in list(dict.fromkeys(list(imgA_glob2loc.keys())+list(imgB_glob2loc.keys()))):
        if k in imgA_glob2loc and k in imgB_glob2loc:
            s[k[0], k[1]] = mfun(
                imgA.data[imgA_glob2loc[k][0], imgA_glob2loc[k][1]], 
                imgB.data[imgB_glob2loc[k][0], imgB_glob2loc[k][1]]
                ) # could be an arbitrary function instead of average
        elif k in imgA_glob2loc:
            s[k[0], k[1]] = imgA.data[imgA_glob2loc[k][0], imgA_glob2loc[k][1]]
        elif k in imgB_glob2loc:
            s[k[0], k[1]] = imgB.data[imgB_glob2loc[k][0], imgB_glob2loc[k][1]]

    if out_path != None:
        cv2.imwrite(out_path, s)
    else:
        return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Coordinate based image stitching')

    # Compute parameters
    parser.add_argument('-i', '--input', type=str, nargs=2, metavar='PATH',
                        required=True, help='Input path of imgA and imgB to be stitched')
    parser.add_argument('-c', '--coord', type=int, nargs=4, metavar='N', required=True,
                        help='Starting point coordinates of the tiles as xA yA xB yB')
    parser.add_argument('-d', '--dim', type=int, nargs=4, metavar='N',
                        required=True, help='Dimension of the tiles as xA yA xB yB')
    parser.add_argument('-o', '--output', default='stitched.png', type=str,
                        metavar='PATH', help='Path to save generated stitch (default: stitch.png)')

    args = parser.parse_args()

    imgA = img(
            cv2.imread(args.input[0]),
            (args.coord[0], args.coord[1]),
            (args.dim[0], args.dim[1])
        )
    imgB = img(
            cv2.imread(args.input[1]),
            (args.coord[2], args.coord[3]),
            (args.dim[2], args.dim[3])
        )
    stitch(imgA, imgB, mfun=average, out_path=args.output)
