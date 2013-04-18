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
  rgb = image.split()
  return (np.array(rgb[0]), np.array(rgb[1]), np.array(rgb[2]))

def to_image_arr(rgba_arrs):
  result = []
  for arr in rgba_arrs:
    result.append(Image.fromarray(arr.astype(np.uint8).copy()))
  return Image.merge("RGB", result)

def to_image(rgba_arrs):
  return to_image_arr(rgba_arrs)
