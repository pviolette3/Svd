import numpy as np
import scipy as sp

class SVDBase:
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
    Return the SVD calculated
    """
    return None

class SVD(SVDBase):

  def __init__(self, arr):
    self.arr = arr
    self.m = arr.shape[0]
    self.n = arr.shape[1]

  def get(self):
    """
    Calculates the SVD of the given matrix.
    returns: The tuple (U, D, V.T) form of SVD, where D is an rxr square matrix
    """
    ata = np.dot(self.arr.T, self.arr)
    eigvals, eigvects = np.linalg.eigh(ata)
    singular_vals, r = self.singular_vals(eigvals)
    print "singular values"
    print singular_vals
    dmat, v = self.get_decomp_matrices(r, singular_vals, eigvals, eigvects)
    print "Inverse should be transpose"
    print np.allclose(np.dot(v.T, v), np.eye(v.T.shape[0]))
    dinv = np.linalg.inv(dmat)
    print "Does I=(D.I V.T A.T) (A V D.I)?"
    print np.allclose(np.eye(r), np.dot(dinv, np.dot(v.T, np.dot(np.dot(self.arr.T, self.arr), np.dot(v, dinv)))))
    print "V eivenvector of A.T A ?"
    print np.allclose(np.dot(ata, v), np.dot(np.dot(v, dmat), dmat))
    print "D inverse works well?"
    print np.allclose(np.dot(np.linalg.inv(dmat), dmat), np.eye(r))
    print "U orthogonal?"
    u = np.dot(self.arr, np.dot(v, dinv))
    print np.allclose(np.dot(u.T, u), np.eye(r))
    return (u, dmat, v.T)
  
  def singular_vals(self, eigenvals):
    dt = np.dtype([('singular_val', np.float64), ('index', np.uint16)])
    singvals = np.zeros(eigenvals.shape, dt)
    i = 0
    for l in np.nditer(eigenvals):
      singvals['singular_val'][i] = l
      singvals['index'][i] = i
      i += 1
    singvals['singular_val'] *= -1
    singsorted = np.sort(singvals, order='singular_val')
    singsorted['singular_val'] *= -1
    r = 0
    for val in np.nditer(singsorted, op_flags=['readwrite']):
      if val['singular_val'] > 0:
        val['singular_val'][...] = np.sqrt(val['singular_val'])
        r += 1
    return (singsorted, r)

  def get_decomp_matrices(self,r, singular_vals, eigenvals, eigvects):
    dmat = np.zeros((r,r), dtype=np.float64)
    v = np.zeros((self.n, r), dtype=np.float64)
    print "v shape is %s" % (v.shape,)
    print "eigenvect shape is %s" % (eigvects.shape,)
    ata = np.dot(self.arr.T, self.arr)
    print "ata shape is %s" % (ata.shape,)
    for i in range(r):
      sv = singular_vals['singular_val'][i]
      dmat[i][i] = sv
      eigv = eigvects[:,singular_vals['index'][i]]
      v[:,i] = eigv
      print np.allclose(np.dot(ata, eigv), sv*sv*eigv)
      print np.allclose(sv * sv, eigenvals[singular_vals['index'][i]])
    print "Done. v shape is now %s" % (v.shape,)
    return (dmat, v)

class NumpyLAPACKSVD(SVDBase):
  def __init__(self, arr):
    self.arr = arr

  def get(self):
    U, D, V = np.linalg.svd(self.arr, full_matrices=False)
    D_full =  np.zeros((U.shape[1], V.shape[0]))
    np.fill_diagonal(D_full, D)
    return U, D_full, V

