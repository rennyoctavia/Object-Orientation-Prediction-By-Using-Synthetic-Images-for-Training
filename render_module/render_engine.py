
import re
import moderngl # For graphics.
import numpy as np # Vectorized math.
from pyquaternion import Quaternion # Rotation.
from PIL import Image # For image files.
from .utilities.math_tools import MathTools # Custom math library.
from .shapes import Shapes
from .mesh import Mesh
from .model import Model

#import pywavefront

class RenderEngine:
	def __init__(self):
		self.time = 0
		self.meshes = []
		self.models = []

		#
		self.cameraPos = np.array([0,-1,0])
		self.cameraOrientation = Quaternion(axis=[1, 0, 0], angle=0)

		self.screenWidth = None
		self.screenHeight = None

		self.pygame = None

		self.ctx = None
	@staticmethod
	def loadFile(fileName, normalize = False):
		import os
		fileEnding = fileName[fileName.rfind('.')+1:]
		if fileEnding != 'obj':
			raise Exception('Files of type ' + fileEnding + ' is not supported. only triangularized obj files are supported.')
		vertices = []
		textureCoords = []
		normals = []
		faces = []
		with open(os.path.dirname(__file__) + '/' + fileName) as file:
			for line in file.readlines():
				if line.startswith('v '):
					vertices.append(np.array([
						float(line.split(' ')[1]),
						float(line.split(' ')[2]),
						float(line.split(' ')[3])
					]))
				elif line.startswith('vt '):
					textureCoords.append(np.array([
						float(line.split(' ')[1]),
						float(line.split(' ')[2]),
					]))
				elif line.startswith('vn '):
					normals.append(np.array([
						float(line.split(' ')[1]),
						float(line.split(' ')[2]),
						float(line.split(' ')[3])
					]))
				elif line.startswith('f '):
					# int(line.split(' ')[1-3 vertex].split('/')[0-2 vert, tex, norm])
					p0 = line.split(' ')[1].split('/')[0]
					p0 = 0 if p0 == '' else int(p0)-1
					t0 = line.split(' ')[1].split('/')[1]
					t0 = 0 if t0 == '' else int(t0)-1
					n0 = line.split(' ')[1].split('/')[2]
					n0 = 0 if n0 == '' else int(n0)-1
					p1 = line.split(' ')[2].split('/')[0]
					p1 = 0 if p1 == '' else int(p1)-1
					t1 = line.split(' ')[2].split('/')[1]
					t1 = 0 if t1 == '' else int(t1)-1
					n1 = line.split(' ')[2].split('/')[2]
					n1 = 0 if n1 == '' else int(n1)-1
					p2 = line.split(' ')[3].split('/')[0]
					p2 = 0 if p2 == '' else int(p2)-1
					t2 = line.split(' ')[3].split('/')[1]
					t2 = 0 if t2 == '' else int(t2)-1
					n2 = line.split(' ')[3].split('/')[2]
					n2 = 0 if n2 == '' else int(n2)-1
					faces.append([
						[
							p0, # First vertex, position.
							t0, # First vertex, texture coordinate.
							n0, # First vertex, normal.
						],
						[
							p1, # Second vertex, position.
							t1, # Second vertex, texture coordinate.
							n1, # Second vertex, normal.
						],
						[
							p2, # Third vertex, position.
							t2, # Third vertex, texture coordinate.
							n2, # Third vertex, normal.
						]
					])
		#
		if normalize:
			center = sum(vertices)/(len(vertices))
			vertices = [v-center for v in vertices]
		#
		vertexData = []
		indices = []
		currentIndex = 0
		seenVertices = {}
		for face in faces:
			for v in face:
				vertInd, texInd, normInd = v
				if (vertInd, texInd, normInd) not in seenVertices:
					ind = currentIndex
					seenVertices[(vertInd, texInd, normInd)] = ind
					currentIndex += 1

					vertexData.append(np.array([
						vertices[vertInd][0],vertices[vertInd][1],vertices[vertInd][2],
						textureCoords[texInd][0],textureCoords[texInd][1],
						normals[normInd][0], normals[normInd][1], normals[normInd][2],
					]))
					indices.append(ind)
				else:
					ind = seenVertices[(vertInd, texInd, normInd)]
					indices.append(ind)
		vertexData = np.array(vertexData)
		indices = np.array(indices)
		
		return vertexData, indices, np.array(vertices)
	@staticmethod
	def loadImageFile(fileName):
		import os
		image = Image.open(os.path.dirname(__file__) + '/' + fileName, mode='r').transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
		return image
	def setDimensions(self, width, height, openWindow = True):
		self.screenWidth = width
		self.screenHeight = height
		if self.pygame is not None:
			self.pygame.display.set_mode((self.screenWidth, self.screenHeight), self.pygame.DOUBLEBUF | self.pygame.OPENGL)
		# import moderngl
		# if openWindow:
		# 	self.ctx = moderngl.create_context()
		# else:
		# 	self.ctx = moderngl.create_standalone_context()
		# self.ctx.enable(moderngl.DEPTH_TEST)
		# self.meshProg = Mesh.getShaderProgram(self.ctx)
	def setPyGame(self,pygame, openWindow = True):
		self.pygame = pygame
		if self.screenWidth is None:
			self.screenWidth = 800
			self.screenHeight = int(600)

		if openWindow:
			self.pygame.init()
			self.pygame.display.set_mode((self.screenWidth, self.screenHeight), self.pygame.DOUBLEBUF | self.pygame.OPENGL)

		import moderngl
		if openWindow:
			self.ctx = moderngl.create_context()
		else:
			self.ctx = moderngl.create_standalone_context()
		self.ctx.enable(moderngl.DEPTH_TEST)
		self.meshProg = Mesh.getShaderProgram(self.ctx)
	def createMesh(self, meshFileName, textureFileName, specularFileName=None):
		newMesh = Mesh()
		newMesh.setShaderProgram(self.meshProg)
		#
		if meshFileName is None:
			vertexData, indices = Shapes.cube()
		else:
			vertexData, indices, vertices = RenderEngine.loadFile(meshFileName)
		#
		texture = RenderEngine.loadImageFile(textureFileName)
		if specularFileName is not None:
			specularMap = RenderEngine.loadImageFile(specularFileName)
		else:
			specularMap = None
		#
		newMesh.loadModel(vertexData, indices, texture, specularMap, self.ctx)
		#self.meshes.append(newMesh)
		return newMesh, vertices
	def createModel(self,mesh):
		newModel = Model()
		newModel.setMesh(mesh)
		self.models.append(newModel)
		return newModel
	def createViewMatrix(self):
		viewMat = MathTools.getTranslation(-self.cameraPos)
		viewMat = (Quaternion(axis=[1, 0, 0], angle=-np.pi*0.5) * self.cameraOrientation.conjugate).transformation_matrix @ viewMat
		return viewMat
	def createProjectionMatrix(self, fov=60/180*np.pi):
		projMat = MathTools.getProjection(fov, self.screenWidth/self.screenHeight, 0.1,100)
		return projMat
	def renderFrame(self, width=800, height=600, fov=60/180*np.pi, showImage=False, lightColor=np.array([1.,1.,1.])):
		self.screenWidth = width
		self.screenHeight = height
		#
		viewMat = self.createViewMatrix()
		projMat = self.createProjectionMatrix(fov=fov)
		#
		self.ctx.clear(0.9, 0.9, 0.9, 1.)
		#
		fbo = self.ctx.simple_framebuffer((self.screenWidth, self.screenHeight))
		fbo.use()
		fbo.clear(0.0, 0.0, 0.0, 1.0)
		for i in range(len(self.models)):
			m = self.models[i]
			#m.orientation *= Quaternion(axis=[1, 2, 3], angle=0.001)
			m.render(self.cameraPos, viewMat, projMat, self.meshProg, lightColor)
		image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
		if showImage:
			image.show()
		colorMatrix = np.array(image)
		return colorMatrix
	def render(self, fov=60/180*np.pi):
		if self.pygame is None:
			raise Exception('Pygame is not yet set.')
		#
		
		# Create view matrix.
		viewMat = self.createViewMatrix()
		# Create Projection matrix.
		projMat = self.createProjectionMatrix(fov)
		# Render.
		## Clear canvas.
		self.ctx.clear(0.0, 0.0, 0.0, 1.)
		#
		for i in range(len(self.models)):
			m = self.models[i]
			#m.orientation = Quaternion(axis=[0, 0, 1], angle=0.01) * m.orientation
			#m.orientation = Quaternion(axis=[np.random.random(), np.random.random(), np.random.random()], angle=(np.random.random()-0.5)*0.4) * m.orientation
			m.render(self.cameraPos, viewMat, projMat, self.meshProg)
		# Swap buffers.
		self.time = self.time + 1







