import unittest
import svd
import numpy as np
from np.testing import assert_array_almost_equal

class TestSVD(unittest.TestCase):

  def test_simple_svd(self):
    test_data = np.array([1, -1, 1, -1, np.sqrt(3), 0]).reshape(3, 2)
    result = np.svd(test_data)
    done = np.dot(np.dot(result[0], result[1]), result[2])
    assert_array_almost_equal(test_data, done, decimal=5)

if __name__ == '__main__':
  unittest.main()
