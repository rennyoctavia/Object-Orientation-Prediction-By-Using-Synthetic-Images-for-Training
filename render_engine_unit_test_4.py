
from render_module.main import RenderModule
import numpy as np

main = RenderModule(True)
main.loadModel('models/09_real_chair/model.obj', 'models/09_real_chair/texture.jpg','models/09_real_chair/specular_map.jpg')
#main.setModelSize(0.1)
main.createBackground(backgroundType=RenderModule.FLAT_BACKGROUND)
#main.createBackground(backgroundType=RenderModule.BOX_BACKGROUND)
main.run()
