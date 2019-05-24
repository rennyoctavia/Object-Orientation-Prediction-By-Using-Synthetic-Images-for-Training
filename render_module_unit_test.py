

import numpy as np
from pyquaternion import Quaternion  # Rotation.

from render_module.main import RenderModule

import matplotlib.pyplot as plt
import time

t0 = time.time()

rm = RenderModule()
# If you want to use a different model from the default. The default is a chair.
#rm.loadModel('models/car_jeep_05/Jeep_Renegade_2016.obj', 'models/car_jeep_05/car_jeep_ren.jpg')
#rm.loadModel('models/07_chair/model.obj', 'models/07_chair/texture.jpg')
rm.loadModel('models/09_real_chair/model.obj', 'models/09_real_chair/texture.jpg','models/09_real_chair/specular_map.jpg')
#rm.loadModel('models/car_jeep_05/Jeep_Renegade_2016.obj', 'models/car_jeep_05/car_jeep_ren.jpg')
#rm.loadModel('models/space_ship_06/space_ship.obj', 'models/space_ship_06/space_ship.png')
rm.setModelSize(0.2)
#rm.createBackground() # Optional if you want to have a background.
rm.createBackground(RenderModule.FLAT_BACKGROUND) # Optional if you want to have a background.
data = rm.getImages(
	numImages = 1,
	width=64,
	height=64,
	showImage=True,
	#quaternion=Quaternion(axis=[1, 2, 3], angle=3), # If not set, vectoriOrientation is used instead.
	#vectorOrientation=np.array([[1,1,0],[0,0,1]]), # If quaternion is set, this parameter is ignored. If quaternion and this is not set, random orientation is used.
	# vectorOrientation=np.array([
	# 	[ 0.76211782, -0.50862014, -0.40060203],
	# 	[ 0.63921295,  0.68942095,  0.34074265]
	# ]),
	# vectorOrientation=np.array([
	# 	[0,1,0.5],
	# 	[0,-0.5,1]
	# ]),
	fov=16/180*np.pi, # Field of view of render. Default is 40 degrees.
	limitAngle=True,
	randomColors=False,
	blur=1,
	noiseLevel=1,
	limitRollAngle=True
)
# print(data['images'])
# print(data['quaternions'])
# print(data['vectorOrientations'])
# print('np.max(data[\'images\'][0])',np.max(data['images'][0]))

# plt.imshow(data['images'][0].astype(np.uint8))
# plt.show()

print('Seconds:', time.time()-t0)