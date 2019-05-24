
from .render_engine import RenderEngine
from pyquaternion import Quaternion # Rotation.
import numpy as np
from .utilities.keyboard import Keyboard
from .utilities.math_tools import MathTools
from PIL import Image # For image files.
from scipy.ndimage.filters import gaussian_filter

class RenderModule:
	BOX_BACKGROUND = 'BOX_BACKGROUND'
	FLAT_BACKGROUND = 'FLAT_BACKGROUND'
	def __init__(self, useAnimatorWindow=False, width=800, height=600):
		
		self.useAnimatorWindow = useAnimatorWindow

		self.moveVel = 0.2

		import pygame as pygame # For outputting to an image.
		self.pygame = pygame

		self.engine = RenderEngine()
		self.engine.screenWidth = width
		self.engine.screenHeight = height
		self.engine.setPyGame(self.pygame, self.useAnimatorWindow)

		self.keyboard = Keyboard()
		self.keyboard.setPyGame(self.pygame)

		self.model = None
		self.modelCenterToFloorShift = None
		self.modelSize = 1
		#mesh = self.engine.createMesh('models/test_chair_04/chair.obj', 'models/test_chair_04/metal.jpg')

		# mesh = self.engine.createMesh('models/test_texture_03/chair.obj', 'models/test_texture_03/wood-texture-2.jpg')
		# self.model = self.engine.createModel(mesh)
		# self.model.orientation = Quaternion(axis=[0, 0, 1], angle=np.pi*0.5) * self.model.orientation
		# self.model.position[2] = 0.79

		self.backgroundInUse = None
		self.flatBackground = None

		self.test = None

		self.isQuit = False
	def loadModel(self, objectFileName, textureFileName, specularFileName=None):
		newMesh, vertices = self.engine.createMesh(objectFileName, textureFileName, specularFileName)
		#self.engine.models = []
		self.model = self.engine.createModel(newMesh)
		self.model.orientation = Quaternion(axis=[1, 0, 0], angle=np.pi*0.5)
		self.modelCenterToFloorShift = np.min(vertices[:,1])
		self.model.position[2] = -self.modelCenterToFloorShift
		self.model.dimmer = 0.7
		#self.model.orientation = Quaternion(axis=[0, 0, 1], angle=np.pi*0.5) * Quaternion(axis=[1, 0, 0], angle=np.pi*0.5) * self.model.orientation
	def createMesh(self, objectFileName, textureFileName):
		newMesh, vertices = self.engine.createMesh(objectFileName, textureFileName)
		# r = np.max(vertices[:,0])
		# l = np.min(vertices[:,0])
		# f = np.max(vertices[:,1])
		# b = np.min(vertices[:,1])
		# u = np.max(vertices[:,2])
		# d = np.min(vertices[:,2])
		return newMesh, vertices
	def orientFLatBackground(self, doRandom=True):
		# Rotate to camera.
		self.flatBackground.orientation = Quaternion() * self.engine.cameraOrientation
		# Translate behind object.
		forward = np.array(self.engine.cameraOrientation.rotate([0,1,0]))
		self.flatBackground.position = forward * 10
		if doRandom:
			# RotateBackground randomly.
			self.flatBackground.orientation = Quaternion(axis=forward,angle=np.pi*2*np.random.random()) * self.flatBackground.orientation
			right = np.array(self.flatBackground.orientation.rotate([1,0,0]))
			up = np.array(self.flatBackground.orientation.rotate([0,0,1]))
			self.flatBackground.position += right*self.flatBackground.size[0] * (np.random.random()-0.5)*1.8
			self.flatBackground.position += up*self.flatBackground.size[2] * (np.random.random()-0.5)*1.8
	def getImages(self, numImages = 1, width=800, height=600, showImage=False, quaternion=None, vectorOrientation=None, fov=40/180*np.pi, limitAngle=False, randomColors=False, blur=0, noiseLevel=0, limitRollAngle=False):
		if self.model is None:
			raise Exception('No model loaded yet. Run "loadModel(...)" before calling this function.')
		images = []
		quaternions = []
		vectorOrientations = []
		for i in range(numImages):
			#
			#self.engine.setDimensions(width, height)
			if quaternion is not None or vectorOrientation is not None:
				if vectorOrientation is not None and quaternion is None:
					quaternion = MathTools.lookAt(vectorOrientation[0],vectorOrientation[1])
				self.model.position = np.array([0.,0.,0.])
				self.model.position[2] = -self.modelCenterToFloorShift * self.modelSize
				self.model.orientation = Quaternion(axis=[1, 0, 0], angle=np.pi*0.5)
				self.engine.cameraOrientation = Quaternion(axis=[1, 0, 0], angle=np.pi*0.5) * quaternion.conjugate
				camPos = np.array([0.,-6.,0.])
				camPos = self.engine.cameraOrientation.rotate(camPos)
				self.engine.cameraPos = camPos
			else:
				camPos = np.array([6.,0.,0.])
				camPos = Quaternion.random().rotate(camPos)
				if limitAngle:
					camPos[2] = np.abs(camPos[2])
					if camPos[2] < 0.2:
						camPos[2] = 0.2
				self.engine.cameraPos = camPos
				up = np.array([0.5-np.random.random(),0.5-np.random.random(),0.5-np.random.random()])
				if limitRollAngle:
					up[2] = np.abs(up[2])
				self.engine.cameraOrientation = MathTools.lookAt(
					-self.engine.cameraPos,
					up
				)

				self.model.position = np.array([0.,0.,0.])
				self.model.position[2] = -self.modelCenterToFloorShift * self.modelSize
				self.model.orientation = Quaternion(axis=[0, 0, 1], angle=np.random.random()*np.pi) * Quaternion(axis=[1, 0, 0], angle=np.pi*0.5)
			self.engine.cameraPos[2] += 2 # Makes the camera point more accurately at the object.
			if self.backgroundInUse == RenderModule.FLAT_BACKGROUND:
				self.orientFLatBackground()
			
			if randomColors:
				t = np.random.random()
				r = np.cos(t*np.pi*2 + 0*2/3*np.pi)*0.5+0.5
				g = np.cos(t*np.pi*2 + 1*2/3*np.pi)*0.5+0.5
				b = np.cos(t*np.pi*2 + 2*2/3*np.pi)*0.5+0.5
				#print('r:',r,'g:',g,'b:',b)
				lightColor = np.array([r,g,b])
			else:
				lightColor = np.array([1.,1.,1.])
			image = self.engine.renderFrame(width, height, fov, showImage=showImage, lightColor=lightColor)
			image = image.astype(np.float)
			# Noise
			if noiseLevel>0:
				noise = np.random.random(image.shape)
				for i in range(1):
					s = 0.01*0
					noise[noise < 0.5] -= s
					noise[noise > 0.5] += s
					b = 1
					rN = gaussian_filter(noise[:,:,0], sigma=b)
					gN = gaussian_filter(noise[:,:,1], sigma=b)
					bN = gaussian_filter(noise[:,:,2], sigma=b)
					noise = np.concatenate((rN.reshape(rN.shape[0],rN.shape[1],1),gN.reshape(gN.shape[0],gN.shape[1],1),bN.reshape(bN.shape[0],bN.shape[1],1)),axis=2)
				#noise *= 255
				image = image + (noise-0.5)*100*noiseLevel
				image[image<0.] = 0.
				image[image>255.] = 255.
			# Blur
			if blur > 0:
				redImg = gaussian_filter(image[:,:,0], sigma=blur)
				greenImg = gaussian_filter(image[:,:,1], sigma=blur)
				blueImg = gaussian_filter(image[:,:,2], sigma=blur)
				image = np.concatenate((redImg.reshape(redImg.shape[0],redImg.shape[1],1),greenImg.reshape(greenImg.shape[0],greenImg.shape[1],1),blueImg.reshape(blueImg.shape[0],blueImg.shape[1],1)),axis=2)
			images.append(image)
			#images.append(noise)
			visOri = self.engine.cameraOrientation.conjugate * self.model.orientation # From the cameras point of view, how does the chair seem to be oriented?
			forward = visOri.rotate([0,1,0])
			up = visOri.rotate([0,0,1])
			vectorOrientations.append(np.array([forward, up]))
			quaternions.append(
				Quaternion(
					w=visOri[0],
					x=visOri[1],
					y=visOri[2],
					z=visOri[3],
				)
			)
		return {'images':images, 'quaternions':quaternions, 'vectorOrientations': vectorOrientations}
	def run(self):
		if not self.useAnimatorWindow:
			raise Exception('Cannot run animation if \'useAnimatorWindow\' property is set to False.')
		while True:
			events = self.pygame.event.get()
			# Detect if window closes.
			for event in events:
				if event.type == self.pygame.QUIT:
					self.pygame.quit()
					self.isQuit = True
					quit()
			# Detect input.
			self.keyboard.update(events)
			
			#
			if self.keyboard.keyEscape.isDown:
				self.pygame.quit()
				quit()

			# Accelerate.
			if self.keyboard.keyC.isDown:
				self.moveVel *= 1.1
			if self.keyboard.keyV.isDown:
				self.moveVel /= 1.1
			
			#
			forward = self.engine.cameraOrientation.rotate(np.array([0,1,0]))
			right = self.engine.cameraOrientation.rotate(np.array([1,0,0]))
			up = self.engine.cameraOrientation.rotate(np.array([0,0,1]))

			# Translate.
			#
			if self.keyboard.keyW.isDown:
				self.engine.cameraPos = self.engine.cameraPos + self.moveVel * forward
			if self.keyboard.keyS.isDown:
				self.engine.cameraPos = self.engine.cameraPos - self.moveVel * forward
			#
			if self.keyboard.keyD.isDown:
				self.engine.cameraPos = self.engine.cameraPos + self.moveVel * right
			if self.keyboard.keyA.isDown:
				self.engine.cameraPos = self.engine.cameraPos - self.moveVel * right
			#
			if self.keyboard.keySpace.isDown:
				self.engine.cameraPos = self.engine.cameraPos + self.moveVel * up
			if self.keyboard.keyLshift.isDown:
				self.engine.cameraPos = self.engine.cameraPos - self.moveVel * up
			# Rotate.
			rotationVel = 0.03
			if self.keyboard.keyRight.isDown:
				self.engine.cameraOrientation = Quaternion(axis=up, angle=-rotationVel) * self.engine.cameraOrientation
			if self.keyboard.keyLeft.isDown:
				self.engine.cameraOrientation = Quaternion(axis=up, angle=rotationVel) * self.engine.cameraOrientation
			if self.keyboard.keyUp.isDown:
				self.engine.cameraOrientation = Quaternion(axis=right, angle=rotationVel) * self.engine.cameraOrientation
			if self.keyboard.keyDown.isDown:
				self.engine.cameraOrientation = Quaternion(axis=right, angle=-rotationVel) * self.engine.cameraOrientation
			if self.keyboard.keyE.isDown:
				self.engine.cameraOrientation = Quaternion(axis=forward, angle=rotationVel) * self.engine.cameraOrientation
			if self.keyboard.keyQ.isDown:
				self.engine.cameraOrientation = Quaternion(axis=forward, angle=-rotationVel) * self.engine.cameraOrientation
			#
			if self.backgroundInUse == RenderModule.FLAT_BACKGROUND:
				self.orientFLatBackground()
			# Render.
			#self.test.orientation = self.test.orientation * Quaternion(axis=[0, 0, 1], angle=0.01)
			self.engine.render()
			self.pygame.display.flip()
			self.pygame.time.wait(6)
	def createBackground(self, backgroundType=None):
		if backgroundType is None:
			backgroundType = RenderModule.BOX_BACKGROUND
		self.backgroundInUse = backgroundType
		if backgroundType == RenderModule.BOX_BACKGROUND:
			# Create floor.
			mesh, vertices = self.engine.createMesh('models/08_floor/model.obj', 'models/08_floor/texture.jpg')
			model = self.engine.createModel(mesh)
			model.orientation = Quaternion(axis=[1, 0, 0], angle=np.pi*0.5)
			model.position[2] = 0
			dist = 7
			model.size *= dist
			# Create walls.
			for i in range(4):
				model = self.engine.createModel(mesh)
				model.orientation = Quaternion(axis=[0, 0, 1], angle=np.pi*0.5 * i)
				model.position[0] = 0 if i==0 or i==2 else dist if i==1 else -dist
				model.position[1] = dist if i==0 else 0 if i==1 or i==3 else -dist
				model.size *= dist
				model.position[2] += dist
		elif backgroundType == RenderModule.FLAT_BACKGROUND:
			backgroundMesh, vertices = self.engine.createMesh('models/10_background/model.obj', 'models/10_background/texture_2.jpg')
			backgroundModel = self.engine.createModel(backgroundMesh)
			backgroundModel.size *= 100
			backgroundModel.size[0] *= 0.75 # Multiply by aspect ratio.
			self.flatBackground = backgroundModel
			#model.orientation =  

		else:
			raise Exception('Background type not recongnized. Expected "RenderModule.BOX_BACKGROUND" or "RenderModule.FLAT_BACKGROUND", but recieved {}'.format(backgroundType))
	def setModelSize(self, size=1):
		if self.model is None:
			raise Exception('No model loaded yet. Run "loadModel(...)" before calling this function.')
		self.modelSize = size
		self.model.size = np.ones(3).astype(float) * self.modelSize
		self.model.position[2] = -self.modelCenterToFloorShift * self.modelSize
	def visualize(self,realFOV,image,positions,dimensions,meshes,realHeights,orientations, changeCoordinate=False):

		targetPos = positions[0]
		# Change coordinates.
		newPositions = []
		newDimensions = []
		if changeCoordinate:
			for i in range(len(positions)):
				p = positions[i]
				d = dimensions[i]
				# Scale.
				newPositions.append(np.array([p[0]/image.shape[1],1-p[1]/image.shape[0]]))
				newDimensions.append(np.array([d[0]/image.shape[1],d[1]/image.shape[0]]))
				# Change center from top left, to center.
				# newPositions[i][0] += newDimensions[i][0] * 0.5
				# newPositions[i][1] += newDimensions[i][1] * 0.5
			positions = newPositions
			dimensions = newDimensions

		if self.isQuit:
			return

		if self.useAnimatorWindow:
			events = self.pygame.event.get()
			# Detect if window closes.
			for event in events:
				if event.type == self.pygame.QUIT:
					self.pygame.quit()
					self.isQuit = True
					return
			self.keyboard.update(events)

		#
		height = int(image.shape[0])
		width = int(image.shape[1])
		aspect = width/height
		#self.engine.setDimensions(width,height)

		self.engine.cameraPos = np.array([0.,0.,0.])
		self.engine.cameraOrientation = Quaternion(axis=[1, 0, 0], angle=0)
		# Create models.
		## Clear models.
		self.engine.models = []
		for i in range(len(meshes)):
			mesh, vertices = meshes[i]
			realHeight = realHeights[i]
			orientation = orientations[i]
			position = positions[i]
			dimension = dimensions[i]
			# Create new model.
			model = self.engine.createModel(mesh)
			# Calculate model height.
			t = np.max(vertices[:,1])
			b = np.min(vertices[:,1])
			height = t-b
			# Re-size model to real height.
			sizeFactor = realHeight / height
			model.size = np.ones(3).astype(float) * sizeFactor
			# Rotate model.
			model.orientation = MathTools.lookAt(orientation[0],orientation[1])
			# Rotate vertices.
			rotMat = model.orientation.rotation_matrix
			rotVertices = (rotMat @ vertices.T).T
			# Calculate distance to object from camera.
			## Calulate observed with of object (orthogonal projection) from camera.
			l = (np.max(rotVertices[:,0]) - np.min(rotVertices[:,0])) * sizeFactor
			## Fraction of with of fov taken by real object from image.
			r = dimension[0]
			##
			distance = l/r/2/np.tan(realFOV/2)
			# Calculate objects position.
			xPos = (position[0]-0.5) * 2*distance*np.tan(realFOV/2)
			zPos = (position[1]-0.5) * 2*distance*np.tan(realFOV/2) / aspect
			yPos = distance
			# Shift model origin to center.
			shift = rotVertices.mean(axis=0)
			xPos -= shift[0]*sizeFactor
			yPos -= shift[1]*sizeFactor
			zPos -= shift[2]*sizeFactor
			#
			model.position = np.array([xPos,yPos,zPos])
			# Rotate the model towards the camera.
			## Pitch.
			angle = np.arctan(model.position[2] / model.position[1])
			model.orientation = Quaternion(axis=[1,0,0],angle=angle) * model.orientation
			## Yaw.
			angle = np.arctan(model.position[0] / model.position[1])
			model.orientation = Quaternion(axis=[0,0,-1],angle=angle) * model.orientation

		if self.useAnimatorWindow:
			self.engine.render(realFOV)
			self.pygame.display.flip()
		else:
			colorMatrix = self.engine.renderFrame(self.engine.screenWidth, self.engine.screenHeight, realFOV,False)
			x = int(targetPos[0])
			y = int(targetPos[1])
			# colorMatrix[:,(x-10):(x+11),:] = 255
			# colorMatrix[(y-10):(y+11),:,:] = 255
			return colorMatrix.astype(np.float)
	def tearDown(self):
		if self.pygame is not None:
			self.pygame.quit()

















