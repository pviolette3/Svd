import numpy as np
import PIL.Image as Image
import PIL.PngImagePlugin

def read_img(filename):
  """
  Read the image from the given directory.
  """
  image = Image.open(filename)
  return png_to_rgba_matrices(image)

def png_to_rgba_matrices(image):
  """
  Input: A PIL *PNG* image object
  Output: A 3xWxH matrix whose elements correspond to the RGB of the image
  (Matix could be 4xWxH if we have an Alpha component)
  """
  r_arr = np.zeros(image.size[0]*image.size[1], dtype=np.uint8)
  g_arr = np.zeros(image.size[0]*image.size[1], dtype=np.uint8)
  b_arr = np.zeros(image.size[0]*image.size[1], dtype=np.uint8)
  a_arr = np.zeros(image.size[0]*image.size[1], dtype=np.uint8)
  i = 0
  for r,g,b,a in image.getdata():
    r_arr[i] = r
    g_arr[i] = g
    b_arr[i] = b
    a_arr[i] = a
    i += 1
  return ([r_arr.reshape(image.size), g_arr.reshape(image.size),
    b_arr.reshape(image.size), a_arr.reshape(image.size)])
 
def to_image_arr(rgba_arrs):
  def place_color(x,y,z):
    print "%s %s %s" % (str(x), str(y), str(z))
    return rgba_arrs[0]
  twod_shape = rgba_arrs[0].shape
  shape = (twod_shape[0], twod_shape[1], 4)
  print shape
  return np.copy(np.fromfunction(place_color, shape, dtype=np.uint8))

def to_image(rgba_arrs):
  return Image.fromarray(to_image_arr(rgba_arrs))
