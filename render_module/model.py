import numpy as np
from pyquaternion import Quaternion # Rotation.

class Model:
	def __init__(self):
		self.position = np.array([0,0,0]).astype(float)
		self.orientation = Quaternion(axis=[1, 0, 0], angle=0)
		self.size = np.ones(3).astype(float)
		self.dimmer = 1
	def setMesh(self,mesh):
		self.mesh = mesh
	def render(self, cameraPos, viewMat, projMat, shaderProgram, lightColor=np.array([1.,1.,1.])):
		if self.mesh is None:
			raise Exception('Cannot render model before mesh is set. Call \"setMesh(someMesh)\" first.')
		self.mesh.position = self.position.copy()
		self.mesh.orientation = Quaternion(self.orientation[0],self.orientation[1],self.orientation[2],self.orientation[3])
		self.mesh.size = self.size.copy()
		self.mesh.dimmer = self.dimmer
		self.mesh.render(cameraPos, viewMat, projMat, shaderProgram, lightColor)
	

