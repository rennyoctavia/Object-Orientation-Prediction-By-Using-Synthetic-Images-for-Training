
from render_module.main import RenderModule
import numpy as np

main = RenderModule(True)
#main.loadModel('models/07_chair/model.obj', 'models/07_chair/texture.jpg')
main.loadModel('models/09_real_chair/model.obj', 'models/09_real_chair/texture.jpg')
#main.loadModel('models/car_jeep_05/Jeep_Renegade_2016.obj', 'models/car_jeep_05/car_jeep_ren.jpg')
#main.loadModel('models/space_ship_06/space_ship.obj', 'models/space_ship_06/space_ship.png')
#main.loadModel('models/08_floor/model.obj', 'models/08_floor/texture.jpg')
main.setModelSize(0.1)
main.createBackground()
main.run()

# data = main.getImages(
# 	numImages = 1,
# 	width=1024,
# 	height=1024,
# 	showImage=True,
# 	#quaternion=Quaternion(axis=[1, 2, 3], angle=1), # If not set, vectoriOrientation is used instead.
# 	#vectorOrientation=np.array([[1,1,0],[0,0,1]]), # If quaternion is set, this parameter is ignored. If quaternion and this is not set, random orientation is used.
# 	fov=60/180*np.pi, # Field of view of render. Default is 40 degrees.
# 	limitAngle=True
# )
# vectors = data['vectorOrientations'][0]
# quaternion = data['quaternions'][0]
# data = main.getImages(
# 	numImages = 1,
# 	width=1024,
# 	height=1024,
# 	showImage=True,
# 	#quaternion=quaternion, # If not set, vectoriOrientation is used instead.
# 	vectorOrientation=vectors, # If quaternion is set, this parameter is ignored. If quaternion and this is not set, random orientation is used.
# 	fov=60/180*np.pi, # Field of view of render. Default is 40 degrees.
# 	limitAngle=True
# )