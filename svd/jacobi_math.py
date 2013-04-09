from numpy import *
import random as rand

# Given a symmetric matrix, returns a good approximation of the diagonalization
# Returns values such that arr[0] = eigenvectors, arr[1] = diagonal matrix w/ eigenvalues, and
# arr[0] * arr[1] * arr[2] ~= the original matrix
def jacobi_diagonalize(sym_arr, offset_threshold=10**-6, record = lambda off,
    cur_diag, cur_steps: None, start = lambda mat: None, stop = lambda res :
    None):
  n =  len(sym_arr)
  diagonal = sym_arr.astype(float).copy()
  steps = eye(n)
  off = offset(diagonal)
  start(diagonal)
  record(off, diagonal, steps)
  while off >= offset_threshold:
    largest_offset_pos = largest_off_index(diagonal)
    diagonalized_small = diagonalize_2by2(small_matr_of(diagonal,
      largest_offset_pos))
    as_n_mat =  promote(diagonalized_small[0],largest_offset_pos, n)
    diagonal = dot(dot(transpose(as_n_mat), diagonal), as_n_mat)
    steps = dot(steps, as_n_mat)
    off = offset(diagonal)
    record(off, diagonal, steps)
  res =  [steps, diagonal, transpose(steps)]
  stop(res)
  return res

def generate_matrix(rows, cols, generating_function):
  result = []
  for i in range(rows):
    for j in range(cols):
      result.append(generating_function(i, j))
  return array(result, dtype=float).reshape(rows, cols)

def random_diagonal_matrix(size, max_element):
  reflected = {}
  def diag_generator(row, col):
    if row <= col:
      val = rand.random() * max_element * (-1)**rand.randint(0,1)
      reflected[(row, col)] = int(val + 1)
      return reflected[(row, col)]
    else:
      return reflected[(col, row)]
  return generate_matrix(size, size, diag_generator)

def largest_off_index(sym_matr):
  max = -1
  pos = [0,0]
  for i in range(len(sym_matr)):
    for j in range(i, len(sym_matr)):
        off = sym_matr[i][j] ** 2
        if off > max and i != j:
          max = off
          pos = [i,j]
  return pos

def diagonalize_2by2(arr):
  eigs = eig_calc(arr)
  vals = eigs[0]
  vects = eigs[1]
  g = array([vects[0], vects[1]])
  d= array([[vals[0], 0], [0, vals[1]]])
  return [transpose(g), d, g]

def eig_calc(arr):
  a = arr[0][0]
  b = arr[0][1]
  c = arr[1][1]
  eigvals = calc_eigvals(a, b, c)
  eigvects = calc_eigvects(a, b, c, eigvals)
  return [eigvals, eigvects]

def calc_eigvals(a, b, c):
  if b == 0:##it is already diagonalized!
    return [a,c]
  eigvals = []
  base = (a + c)/2
  radius = sqrt(a**2 - 2*a*c + 4*b**2 + c**2)/2
  eigvals.append(base + radius)
  eigvals.append(base - radius)
  return eigvals

def calc_eigvects(a, b, c, eigvals):
  if b == 0:##already diagonal...return I
    if eigvals[0] == a:
       return [array([1.0,0.0]), array([0.0, 1.0])]
    elif eigvals[0] == b:
      return  [array([0.0,1.0]), array([1.0, 0.0])]
    else:
      raise err
  def eigvect_of(l):
    v1 = 1
    v2 = (l - a) / b
    mag = sqrt(v1**2 + v2**2)
    return array([v1 / mag, v2 / mag]) 
  return map(eigvect_of, eigvals)


def small_matr_of(large, pos):
  a = pos[0]
  b = pos[1]
  return array([large[a][a], large[a][b], large[b][a],
    large[b][b]]).reshape(2,2)
class OffFinder:
  def __init__(self):
    self.pos = [0,0]
  
  def largest_off_index(self, sym_matr):
    pos = self.pos
    if pos[1] >= len(sym_matr) - 1:
      if pos[0] >= len(sym_matr) - 2:
        pos[0] = 0
      else:
        pos[0] = pos[0] + 1
      pos[1] = pos[0] + 1 # Not on diagonal
    else:
      pos[1] = pos[1] + 1
    return pos

def promote(two_by_two, pos, up_dim):
  result = eye(up_dim)
  a = pos[0]
  b = pos[1]
  result[a][a] = two_by_two[0][0]
  result[a][b] = two_by_two[0][1]
  result[b][a] = two_by_two[1][0]
  result[b][b] = two_by_two[1][1]
  return result

def offset(matr):
  res = 0
  for i in range(len(matr)):
    for j in range(len(matr)):
      if i != j:
        res += matr[i][j] ** 2
  return res
