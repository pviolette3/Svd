import numpy as np
import Image
import np.linalg.eig as npeig

def compress_svd(arr, goal_proportion=0.5):
  """
  Give a best rank approximation based on a goal proportion
  of the original image
  """
  arr = svd(arr)
  total = np.sum(svd_arr[1])
  i = 0
  for val in 

def svd(matr):
  """
  Calculates the SVD of the given matrix.
  returns: [U, D, V.T] form of SVD, where D is an rxr square matrix
  """
  ata = np.dot(matr.T, matr)
  eigvals, eigvects = npeig(ata)#A^T A
  res = singular_vals(combine(eigvals, eigvects))
  v = res['eigvect']
  d = res['eigvals']
  u = np.zeros(res.shape, np.float64)
  i = 0
  for vals in np.nditer(u, op_flags['readwrite']):
    u[...] = np.dot(matr, v[i]) / d[i]
  return np.array([u, d, v.T])

def combine(eigvals, eigvects):
  """
  Combines eigvals and eigvects into a single multidimesional array
  returns: res['eigval'] = eigvals and res['eigvect'] = eigvects
  """
  dt = np.dtype([('eigval', np.float64), ('eigvect', np.float64, (eigvects.shape[1],))])
  result = np.zeros(eigvals.shape, dt)
  i = 0 # we'll iterate in order
  for l in np.nditer(eigvals, order='K'):
    result['eigval'][i] = l
    result['eigvect'][i] = eigvects[i]
    i += 1
  return result

def singular_vals(eig_array):
  eig_sorted = np.sort(eig_array, order='eigval')
  eigvals = eig_sorted['eigval'][::-1] #sort ascending
  for l in np.nditer(eigvals, op_flags['readwrite']):
    eigvals[...] = np.sqrt(l)
  return eig_sorted


def best_rank(img_svd):


def process(double_img_arr):
