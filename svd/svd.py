import numpy as np
import scipy as sp

class SVDBase:
  def best_rank(self, to_rank):
    """
    Gives the matrix that is the best (to_rank) rank approximation of the
    given matrix.
    """
    u, d, v_t = best_rank_svd(to_rank)
    return np.dot(np.dot(u, d), v_t)

  def best_rank_svd(self, to_rank):
    """
    Give a best rank approximation svd of the matrix arr of the given rank.
    Returns the tuple (u', d', v_t') such that the product has rank to_rank
    """
    u, d, v_t = self.get()
    u_prime = np.zeros((u.shape[0], to_rank) )
    d_prime = np.zeros((to_rank, to_rank))
    v_t_prime = np.zeros((to_rank, v_t.shape[1]))
    for i in range(to_rank):
      u_prime[..., i] = u[..., i] #copy col
      d_prime[i][i] = d[i][i]
      v_t_prime[i,...] = v_t[i,...] #copy row
    return (u_prime, d_prime, v_t_prime)

  def get():
    """
    Return the SVD calculated
    """
    return None

class SVD(SVDBase):

  def __init__(self, arr):
    SVDBase.__init__();
    self.arr = arr

  def get(self):
    """
    Calculates the SVD of the given matrix.
    returns: The tuple (U, D, V.T) form of SVD, where D is an rxr square matrix
    """
    ata = np.dot(matr.T, matr)
    eigvals, eigvects = np.linalg.eigh(ata)
    res = singular_vals(combine(eigvals, eigvects))
    v_t = res['eigvect']
    d = res['eigval']
    u, d_as_mat = solve_for_u(d, v_t, matr) 
    return (u, d_as_mat, v_t)
   
  def solve_for_u(self, d, v_t):
    u = np.zeros((matr.shape[0], d.shape[0]), np.float64)
    d_as_mat = np.zeros((d.shape[0], d.shape[0]), np.float64)
    for i in range(d.shape[0]):
      av = np.dot(matr, v_t[i].T)
      res = av / d[i]
      u[..., i] =  res
      d_as_mat[i][i] = d[i]
    return (u, d_as_mat)

  def combine(eigvals, eigvects):
    """
    Combines eigvals and eigvects into a single multidimesional array
    returns: res['eigval'] = eigvals and res['eigvect'] = eigvects
    """
    dt = np.dtype([('eigval', np.float64), ('eigvect', np.float64, (eigvects.shape[1],))])
    result = np.zeros(eigvals.shape, dt)
    i = 0 # we'll iterate in order
    for l in np.nditer(eigvals, order='K'):
      result['eigval'][i] = eigvals[i]
      result['eigvect'][i] = eigvects[i]
      i += 1
    return result

  def singular_vals(eig_array):
    """
    Sorts the eigenvalues in ascending order, then finds
    the singular values by taking the square roots
    """
    eig_array['eigval'] *= -1
    eig_sorted = np.sort(eig_array, order='eigval')
    eig_sorted['eigval'] *= -1
    r = 0 
    for val in np.nditer(eig_sorted['eigval'], op_flags=['readwrite']):
      if val > 10**-6:
        val[...] = np.sqrt(val)
        r += 1
    result = np.zeros(r, dtype=eig_sorted.dtype)
    for i in range(r):
      result[i] = eig_sorted[i]
    return result

class NumpyLAPACKSVD(SVDBase):
  def __init__(self, arr):
#    SVDBase.__init__(self)
    self.arr = arr

  def get(self):
    U, D, V = np.linalg.svd(self.arr, full_matrices=False)
    D_full =  np.zeros((U.shape[1], V.shape[0]))
    np.fill_diagonal(D_full, D)
    return U, D_full, V

