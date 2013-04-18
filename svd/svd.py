import numpy as np
import scipy as sp

class SVDBase:
  def __init__(self, arr):
    """
    Takes in an array to compute the SVD.
    """
    self.arr = arr

   
  def best_rank(self, to_rank):
    """
    Gives the matrix that is the best (to_rank) rank approximation of the
    given matrix.
    """
    u, d, v_t = self.best_rank_svd(to_rank)
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
    for i in range(min(to_rank, d.shape[0])):
      u_prime[..., i] = u[..., i] #copy col
      d_prime[i][i] = d[i][i]
      v_t_prime[i,...] = v_t[i,...] #copy row
    return (u_prime, d_prime, v_t_prime)

  def get():
    """
    Return the SVD for the given matrix. Subclasses must override.
    """
    return None

class SVD(SVDBase):

  def __init__(self, arr):
    SVDBase.__init__(self, arr)
    self.m = arr.shape[0]
    self.n = arr.shape[1]

  def get(self):
    """
    Calculates the reduced form SVD of the given matrix.
    returns: The tuple (U, D, V.T) form of SVD, where D is an rxr square matrix.
    """
    ata = np.dot(self.arr.T, self.arr)
    eigvals, eigvects = np.linalg.eigh(ata)
    singular_vals, r = self.singular_vals(eigvals)
    dmat, v = self.get_decomp_matrices(r, singular_vals, eigvects)
    dinv = np.linalg.inv(dmat)
    u = np.dot(self.arr, np.dot(v, dinv))
    return (u, dmat, v.T)
  
  def singular_vals(self, eigenvals):
    """
    Converts eigenvalues into singular values. Stores the original index for joins
    with corresponding eigenvectors.
    Returns: result such that result['singular_val'] is the singular value and result['index'] is the index of the corresponding eigenvector of A.T A. NOTE: This is not a full rxr matrix.
    """
    singvals = self.combine_with_index(eigenvals)
    singsorted = self.sort_singular_vals(singvals)
    r = 0
    for val in np.nditer(singsorted, op_flags=['readwrite']):
      if val['singular_val'] > 0:#only get non-zero or greater
        val['singular_val'][...] = np.sqrt(val['singular_val'])
        r += 1
    return (singsorted, r)

  def sort_singular_vals(singvals):
    singvals['singular_val'] *= -1
    singsorted = np.sort(singvals, order='singular_val')
    singsorted['singular_val'] *= -1
    return singsorted
  
  def combine_with_index(self, eigenvals):
    """
    Helper method that saves the index of eigenvalues after sorting, to be used
    to join with corresponding eigenvalues.
    """
    dt = np.dtype([('singular_val', np.float64), ('index', np.uint16)])
    singvals = np.zeros(eigenvals.shape, dt)
    i = 0
    for l in np.nditer(eigenvals):
      singvals['singular_val'][i] = l
      singvals['index'][i] = i
      i += 1
    return singvals

  def get_decomp_matrices(self,r, singular_vals, eigvects):
    """
    Converts matrices of singular values and eigenvectors into D and V, respectively.
    D is an rxr matrix, and V is an mxr matrix, where r is the number of non-zero singular values.

    Returns: (D, V)
    """
    dmat = np.zeros((r,r), dtype=np.float64)
    v = np.zeros((self.n, r), dtype=np.float64)
    ata = np.dot(self.arr.T, self.arr)
    for i in range(r):
      sv = singular_vals['singular_val'][i]
      dmat[i][i] = sv
      eigv = eigvects[:,singular_vals['index'][i]]
      v[:,i] = eigv
    return (dmat, v)

class NumpyLAPACKSVD(SVDBase):
  def __init__(self, arr):
    SVDBase.__init__(self, arr)

  def get(self):
    """
    Returns the SVD composition calculated by the numpy svd function, which uses FORTRAN
    like a boss.
    """
    U, D, V = np.linalg.svd(self.arr, full_matrices=False)
    D_full =  np.zeros((U.shape[1], V.shape[0]))
    np.fill_diagonal(D_full, D)
    return U, D_full, V

