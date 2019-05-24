
from render_module.main import RenderModule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyquaternion import Quaternion

main = RenderModule(False,int(3024*0.1*10), int(4032*0.1*10))

image = np.array(main.engine.loadImageFile('models/09_real_chair/right.jpg'))
positions = np.array([[0.5,0.5],[0.4,0.6],[0.8,0.7]])
positions[:,0] *= 3024
positions[:,1] *= 4032
dimensions = np.array([[0.99,0.9],[0.2,0.2],[0.25,0.25]])
dimensions[:,0] *= 3024
dimensions[:,1] *= 4032

meshA = main.createMesh('models/09_real_chair/model.obj', 'models/09_real_chair/texture.jpg')
meshB = main.createMesh('models/07_chair/model.obj', 'models/07_chair/texture.jpg')
meshes = [meshA, meshB, meshB]
realHeights = [1,2,1]
orientations = [
	np.array([
		[ 0.76211782, -0.50862014, -0.40060203],
		[ 0.63921295,  0.68942095,  0.34074265]
	]),
	np.array([
		[-0.43853511, -0.22802068,  0.86930635],
		[ 0.69653782, -0.6974711 ,  0.16843137]
	]),
	np.array([
		[ 0.48257419, -0.34863079, -0.80347914],
		[ 0.8756424 ,  0.17182414,  0.45136111]
	])
]

# realFOV,image,positions,dimensions,meshes,realHeights,orientations
#fig = plt.figure()
#pltImage = plt.imshow(np.zeros((int(3024*0.1), int(4032*0.1),3)))
#plt.show()
#plt.plot([1,2,3])
while True:
	result = main.visualize(40/180*np.pi,image, positions, dimensions, meshes, realHeights, orientations,True)
	plt.imshow(result.astype(np.uint8))
	#pltImage.set_data(image)
	plt.show()
	#plt.draw()
	print('new frame')
	#print(np.random.random(),main.keyboard.keyEscape.isDown)

	# orientations[0] = np.array([
	# 	Quaternion.random().rotate([0,1,0]),
	# 	Quaternion.random().rotate([0,0,1])
	# ])
	break

	if main.isQuit or main.keyboard.keyEscape.isDown:
		main.tearDown()
		quit()
	main.pygame.time.wait(6)




