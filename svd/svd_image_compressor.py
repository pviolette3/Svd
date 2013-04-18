import svd
import image_lib
import numpy as np
import Image
from svd import SVD, NumpyLAPACKSVD

from scipy import ndimage

def process(filename):
  image = compress(filename)
  image.save("comp_"+filename)
  return image

def compress(filename):
  """
  Given a png filename, this will use svd to find the best
  rank approximation of 50% of the rank of the given png.

  It returns the image calculated.
  """
  colors = image_lib.read_img(filename) 
  approx = []
  approximator = RankApprox(1.0)
  for arr in colors:
    print arr.shape
    U, D, V = NumpyLAPACKSVD(arr).best_rank_svd(arr.shape[1] / 10)
    approx.append(np.dot(U, np.dot(D, V)))
  for i in range(len(approx)):
    approx[i] = to_uint8(approx[i])
  return image_lib.to_image(approx)

def to_uint8(float_arr):
  for x in np.nditer(float_arr, op_flags=['readwrite']):
    if x > 255:
      x[...] = 255
    elif x < 0:
      x = 0
    else:
      x[...] = np.floor(x)
  return float_arr.astype(np.uint8)

def through_the_pipes(filename):
  colors = image_lib.read_img(filename)
  image_lib.to_image(map(to_uint8,colors)).save("pipes_" + filename)

class RankApprox:
  
  def __init__(self, ratio):
    self.ratio = ratio

  def approx(self, arr):
    goal_rank = int(self.ratio * np.rank(arr))
    return svd.best_rank(arr, goal_rank)

