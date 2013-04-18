import svd
import image_lib
import numpy as np
import Image
from svd import SVD, NumpyLAPACKSVD

from scipy import ndimage

def process(filename, rank_comp=10):
  image = compress(filename, rank_comp)
  image.save("comp_"+filename)
  return image

def compress(filename, rank_comp):
  """
  Given a png filename, this will use svd to find the best
  rank approximation of 10% of the rank of the given png.

  It returns the image calculated.
  """
  colors = image_lib.read_img(filename) 
  approx = []
  for arr in colors:
    print arr.shape
    val = SVD(arr).best_rank(arr.shape[1] / rank_comp)
    approx.append(val)
  for i in range(len(approx)):
    approx[i] = to_uint8(approx[i])
  return image_lib.to_image(approx)

def to_uint8(float_arr):
  for x in np.nditer(float_arr, op_flags=['readwrite']):
    if x > 255:
      x[...] = 255
    elif x < 0:
      x[...] = 0
    else:
      x[...] = np.floor(x)
  return float_arr.astype(np.uint8)
