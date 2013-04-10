import svd
import image_lib
import numpy as np
import Image

def process(filename):
  compress(filename).save("comp_"+filename)
  return None

def compress(filename):
  """
  Given a png filename, this will use svd to find the best
  rank approximation of 50% of the rank of the given png.

  It returns the image calculated.
  """
  colors = image_lib.read_img(filename) 
  approx = []
  approximator = RankApprox(0.5)
# for arr in colors:
#   for x in np.nditer(arr, op_flags=['readwrite']):
#     x[...] = x / 255.0 #normalize the array for svd purposes
#    approx.append( approximator.approx(arr) )
#  for a in approx:
#    for x in np.nditer(a, op_flags=['readwrite']):
#      x[...] = x * 255
#  return image_lib.to_image(approx)
  return Image.merge("RGB", colors)

class RankApprox:
  
  def __init__(self, ratio):
    self.ratio = ratio

  def approx(self, arr):
    goal_rank = int(self.ratio * np.rank(arr))
    return svd.best_rank(arr, goal_rank)

