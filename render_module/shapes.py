
import numpy as np

class Shapes:
	def __init__(self):
		pass
	@staticmethod
	def cube():
		return np.array([
			#  x, y, z, r, g, b
			1,-1,-1, np.random.rand(), np.random.rand(), np.random.rand(),
			1, 1,-1, np.random.rand(), np.random.rand(), np.random.rand(),
			-1, 1,-1, np.random.rand(), np.random.rand(), np.random.rand(),
			-1,-1,-1, np.random.rand(), np.random.rand(), np.random.rand(),
			1,-1, 1, np.random.rand(), np.random.rand(), np.random.rand(),
			1, 1, 1, np.random.rand(), np.random.rand(), np.random.rand(),
			-1, 1, 1, np.random.rand(), np.random.rand(), np.random.rand(),
			-1,-1, 1, np.random.rand(), np.random.rand(), np.random.rand()
		]), np.array([
			0,2,1, # Bottom.
			0,3,2,
			0,1,4, # Right.
			1,5,4,
			3,0,4, # Front.
			3,4,7,
			3,7,6, # Left.
			3,6,2,
			1,2,5, # Back.
			2,6,5,
			4,5,6, # Top.
			4,6,7,
		])

#print(Shapes.cube())







