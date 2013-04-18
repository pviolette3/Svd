import unittest
import svd
import numpy as np
from numpy.testing import assert_array_almost_equal

class TestSVD(unittest.TestCase):

  def setUp(self):
    svd.TEST = False

  def test_simple_svd(self):
    test_data = np.array([1, -1, 1, -1, 3, 0], dtype=np.float64).reshape(3, 2)
    result = svd.svd(test_data)
    done = np.dot(np.dot(result[0], result[1]), result[2])
    assert_array_almost_equal(test_data, done, decimal=10)

  def test_factorization(self):
    r2 = np.sqrt(2)
    u = np.array([[1/r2, 1/r2], [-1/r2, 1/r2]])
    d = np.array([[4, 0],[0, 1]])
    v_t = np.eye(2)
    mat = np.dot(np.dot(u, d), v_t)
    result = svd.svd(mat)
    assert_array_almost_equal(mat, np.dot(np.dot(result[0], result[1]), result[2]))
    assert_array_almost_equal(u, result[0], decimal=10)
    assert_array_almost_equal(d, result[1], decimal=10)
    assert_array_almost_equal(v_t, result[2], decimal=10)

  def test_4by2_factorization(self):
    r3 = np.sqrt(3)
    r2 = np.sqrt(2)
    u = np.array([[1.0/4.0, 1.0/4.0, 1.0/4.0],
                  [1.0/4.0, -1.0/4.0, 1.0/4.0 ],
                  [1.0/4.0, 1.0/4.0, -1.0/4.0],
                  [1.0/4.0, -1.0/4.0, -1.0/4.0]])
    d = np.zeros((3,3), dtype=np.float64)
    d[0][0] = 10
    d[1][1] = 5
    d[2][2] = 2
    v = np.array([[1/r3, 1/r3, 1/r3] ,[1/r2, -1/r2, 0]])
    mat = np.dot(np.dot(u, d), v.T)
    result = svd.svd(mat)
    assert_array_almost_equal(mat, np.dot(np.dot(result[0], result[1]), result[2]))

  def test_2by4_factorization(self):
    svd.TEST = True
    r3 = np.sqrt(3)
    r2 = np.sqrt(2)
    u = np.array([[1.0/4.0, 1.0/4.0, 1.0/4.0],
                  [1.0/4.0, -1.0/4.0, 1.0/4.0 ],
                  [1.0/4.0, 1.0/4.0, -1.0/4.0],
                  [1.0/4.0, -1.0/4.0, -1.0/4.0]])
    d = np.zeros((3,3), dtype=np.float64)
    d[0][0] = 10
    d[1][1] = 5
    d[2][2] = 2
    v = np.array([[1/r3, 1/r3, 1/r3] ,[1/r2, -1/r2, 0]])
    #Now we gon transpose it
    mat = np.dot(np.dot(v, d), u.T)
    result = svd.svd(mat)
    trans_res = svd.svd(mat.T)
    assert_array_almost_equal(result[0].T, trans_res[2])
    assert_array_almost_equal(mat, np.dot(np.dot(result[0], result[1]), result[2]))
'''  def test_best_rank_approx(self):
     A = [[0,0,0,0,0,0,0 ],
       [ 0,0,0,1,0,0,0 ],
       [ 0,0,1,1,1,0,0 ],
       [ 0,0,1,0,1,0,0 ],
       [ 0,1,0,0,0,1,0 ],
       [ 0,1,1,1,1,1,0 ],
       [ 0,1,1,1,1,1,0 ],
       [ 0,1,0,0,0,1,0 ],
       [ 0,1,0,0,0,1,0 ],
       [ 0,0,0,0,0,0,0 ]]
     rank_2 = [[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0 ],
      [ 0.0,0.0182,0.3146,0.2733,0.3146,0.0182,0.0 ],
      [ 0.0,0.0024,1.0422,0.9025,1.0422,0.0024,0.0 ],
      [ 0.0,0.0158,0.7276,0.6292,0.7276,0.0158,0.0 ],
      [ 0.0,0.9991,0.0158,0.0364,0.0158,0.9991,0.0 ],
      [ 0.0,0.9991,0.0158,0.0364,0.0158,0.9991,0.0 ],
      [ 0.0,1.0015,1.0264,0.9390,1.0264,1.0015,0.0 ],
      [ 0.0,1.0015,1.0264,0.9390,1.0264,1.0015,0.0 ],
      [ 0.0,0.9991,0.0158,0.0364,0.0158,0.9991,0.0 ],
      [ 0.0,0.9991,0.0158,0.0364,0.0158,0.9991,0.0 ]]
     a_arr = np.array(A, dtype=np.float64)
     rank_2_approx = np.array(rank_2, dtype=np.float64)
     self.assertEqual(rank_2_approx.shape, a_arr.shape)
     approx = svd.best_rank(a_arr, 2)
     self.assertEqual(a_arr.shape, approx.shape)
     assert_array_almost_equal(approx, rank_2_approx, 2)
'''
if __name__ == '__main__':
  unittest.main()
