def test_data():
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
  return mat
