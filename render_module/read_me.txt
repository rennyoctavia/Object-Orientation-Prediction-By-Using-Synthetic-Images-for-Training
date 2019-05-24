

This file describes how to use the render module.

Changelog:
	-version-yyyy-mm-dd-
	v_00: 2019.03.27:
		Better graphics.
	v_01: 2019.03.27:
		Slightly better graphics. New texture amongst other things.
		The ability to choose orientation of generated images.
		Updated unit_test.py with new functionality.
	v_02: 2019.03.29:
		Custom file loading. see "loadModel(...)" method.
	v_03: 2019.04.1:
		Can use forward and up vectors to describe orientation.
	v_04: 2019.04.2:
		Can adjust field of view in getImages function. New chair model is added.
	v_05: 2019.04.2:
		Changed default fov.
	v_06: 2019.04.8:
		Background added.
	v_07: 2019.04.11:
		New model added.
		Ability to choose size of model.
	v_08: 2019.04.12:
		visualization added.
	v_08: 2019.04.15:
		Model made darker.
		Alternative background.
	v_08: 2019.04.16:
		Limit roll angle.
		Added specular mapping.
Content:
	See unit_test.py for example of use.
	Dependencies:
		pyquaternion:
			Installation:
				pip install pyquaternion
		moderngl:
			Installation:
				pip install moderngl
	How to import?
		from main import RenderModule
	How to initialize the render module:
			rm = RenderModule()
		The signature:
			RenderModule RenderModule(useAnimatorWindow=False)
		'useAnimatorWindow' is False by default. If it is true, a pygame window will be created upon initialization of the render module.
	How to get the images. The getImages(...) function:
		Description:
			The function generates images of chairs with random orientation (random as default (but can be overrided)).
		The signature:
			dict getImages(numImages = 1, width=800, height=600, showImage=False, quaternion=None, vectorOrientation=None, fov=40/180*np.pi, limitAngle=False, randomColors=False, blur=0, noiseLevel=0, limitRollAngle=False)
		Return value:
			The returned argument is a dictionary with keys 'images', 'quaternions', and 'vectorOrientations'.
			The value to the 'images' key is a list of matrices. The dimensions are pixel width, height and 3. 3 are the red, green, and blue color channels.
			The value of the 'quaternions' key is a list of Quaternion objects describing the orientation of the rendered images.
			The value of the 'vectorOrientations' key is a list of 2 by 3 "numpy.ndarray"s. Two 3 dimentional vectors describe the forward and uppwards direction of the model that is rendered.
			The images, quaternions, and vectorOrientations at the same indices in their respective lists, are associated.
		Properties:
			"numImages" is the number of images to be generated.
			"width" and "height" is the dimensions of the generated images.
			"showImage" is a boolean that chooses wether the images generated will be displayed or not.
			"quaternion" is an optional argument. If not set, the vectorOrientation is used instead.
			"vectorOrientation" is a 2 by 3 numpy.ndarray. It is two 3 dimentional vectors, the first one describing the forward direction of the model, the second one is the models uppwards direction. If the 'quaternion' parameter is set, this parameter is ignored. If both the 'quaternion' parameter and the 'vectorOrientation' parameter is not set, the generated images will have a random orientation.
			"fov" is field of view. A low number will give a zoomed in effect. The unit is radians. To convert from degrees to radians, multiply with pi and divide by 180. Default is 60 degrees.
			"limitAngle" is a boolean that determines whether the object only will be rendered from above or not.
			"randomColors" is whether the light color should be random or not.
			"blur" is how blurry the image should be. 0 in no blur.
			"noiseLevel" is the noise level in the generated images.
			"limitRollAngle" is a boolean that sais wheter the you should exclude images that are upside down. If quaternion or vectorOrientation is set, this parameter is excluded.
	How to load custom models:
		Signature:
			rm.loadModel(self, objectFileName, textureFileName)
		Example:
			rm.loadModel('render_engine/models/car_jeep_04/Jeep_Renegade_2016.obj', 'render_engine/models/car_jeep_04/car_jeep_ren.jpg')
		What are the paths to the different models.
				'models/car_jeep_05/Jeep_Renegade_2016.obj', 'models/car_jeep_05/car_jeep_ren.jpg'
				'models/07_chair/model.obj', 'models/07_chair/texture.jpg'
				'models/09_real_chair/model.obj', 'models/09_real_chair/texture.jpg' # This model is large and should be re-sized to 0.1.
				'models/space_ship_06/space_ship.obj', 'models/space_ship_06/space_ship.png' # This model is large and should be re-sized to 0.3.
			Some of the models are large and should be re-sized. See "How to change model size?" in this document.
	How to set background:
			rm.createBackground() # Run this before the getImages(...) function.
		When using background, the limitAngle argument to the getImages(...) function should be true to prevent images from below the background.
		Alternatively a different background could be used.
			rm.createBackground(RenderModule.BOX_BACKGROUND) # Run this before the getImages(...) function.
		The options are:
			RenderModule.BOX_BACKGROUND,
			RenderModule.FLAT_BACKGROUND.
			
	How to change model size?
		Run this code after loading the model using "rm.loadModel(...)".
			rm.setModelSize(someSize)
		"someSize" is some floating point. Could be 0.1 if you want the model to be 10% if the original size.
	How to visualize images?
		First, import render module and numpy.
			from render_module.main import RenderModule
			import numpy as np
		Initialize the render module.
				main = RenderModule(True,int(3024*0.1), int(4032*0.1))
		The RenderModule init signature is:
			RenderModule RenderModule(useAnimatorWindow=False, width=600, height=800)
		You want the "useAnimatorWindow" to be true, so that you can see the window with the output. The height and width is the size in pixels of the input. In the example we multiply width 0.1, to make the window smaller for practical purposes.
		The next step is to load the meshes of the different possible models. Only do this once as it requires a lot of performace.
			meshA = main.createMesh('models/09_real_chair/model.obj', 'models/09_real_chair/texture.jpg')
			meshB = main.createMesh('models/07_chair/model.obj', 'models/07_chair/texture.jpg')
		Put the meshes in a list in the same order as you want them to be rendered.
			meshes = [meshA, meshB, meshA]
		In this example we render 3 objects. We use meshA twice.
		In this example we hardcode the positions where the objects should be rendered.
			positions = np.array([[0.1,0.5],[0.4,0.6],[0.8,0.7]])
		The numbers represent where on the screen in relative coordinates where the objects should be rendered. 0 is left side of the screen. 1 is to the right. The first number is the horizontal coordinate, the second number is the vertical one.
		We now set the dimensions. It is a list of widths and heights.
			dimensions = np.array([[0.1,0.9],[0.2,0.2],[0.25,0.25]])
		We give the real sizes of the physical objects. The units are meters.
			realHeights = [1,2,1]
		We also need the orientations of the rendered objects. This is some hardcoded orientations.
			orientations = [
				np.array([
					[-0.43853511, -0.22802068,  0.86930635],
					[ 0.69653782, -0.6974711 ,  0.16843137]
				]),
				np.array([
					[ 0.76211782, -0.50862014, -0.40060203],
					[ 0.63921295,  0.68942095,  0.34074265]
				]),
				np.array([
					[ 0.48257419, -0.34863079, -0.80347914],
					[ 0.8756424 ,  0.17182414,  0.45136111]
				])
			]
		The orientations are given as a pair of 3d vectors. The forwards direction and the uppwards direction. The vectors does not have to have a unit length. The angle does not have to be 90 degrees, as a none right angle will be corrected for. The angle should not be 0 or 180 degrees (the forwards and uppwards direction should not be parallel)
		The next step is to animate the models.
			while True:
				main.visualize(90/180*np.pi,image, positions, dimensions, meshes, realHeights, orientations)
				#print(np.random.random(),main.keyboard.keyEscape.isDown)
				if main.isQuit or main.keyboard.keyEscape.isDown:
					main.tearDown()
					quit()
				main.pygame.time.wait(6)
		The signature of the visualize function is:
			visualize(realFOV,image,positions,dimensions,meshes,realHeights,orientations)
			"realFOV" is the field of view of the images. The units are radians.
			"image" is a numpy.ndarray. Its dimensions are (height is pixels) by (width in number of pixels) by (3 color channels (red green blue)). The values of the matrix is between 0 and 1.
			"positions" is a list or ndarray is 2d vectors. The values are between 0 and one depending on where in the image the objects should be rendered.
			"dimensions" is a list or ndarray is 2d vectors. The values are width and height relative to the total width and height of the image.
			"meshes" are the meshes of the models to be rendered as explained in the example above.
			"realHeights" are the real heights of the objects to be rendered in meter units. You can look at the example above.
			"orientations" are the orientations of the models. See example above.
		You can also see the "render_engine_unit_test_3.py" For a working example.