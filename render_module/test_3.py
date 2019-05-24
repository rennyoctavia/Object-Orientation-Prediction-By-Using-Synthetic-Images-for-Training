import moderngl # For graphics.
import numpy as np # Vectorized math.
from pyquaternion import Quaternion # Rotation.
from PIL import Image # For image files.
from utilities.math_tools import Matrices as mt # Custom math library.
import pygame # For outputting to an image.
from pygame.locals import DOUBLEBUF, OPENGL

pygame.init()
pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)

#ctx = moderngl.create_standalone_context()
ctx = moderngl.create_context()
ctx.enable(moderngl.DEPTH_TEST)

prog = ctx.program(
    vertex_shader='''
		#version 330

		in vec3 pos;
		in vec3 in_color;

		uniform mat4 model;
		uniform mat4 view;
		uniform mat4 proj;

		out vec3 v_color;

		bool isEq(float a, float b, float margin);

		void main() {
			in_color;
			proj;

			//v_color = in_color;
			vec4 localPos = vec4(pos, 1.0);
			v_color = localPos.xyz + 0.00000001*in_color;
			gl_Position = proj * view * model * localPos;
		}
		bool isEq (float a, float b, float margin) {
			return abs(a-b) < margin;
		}
	''',
	fragment_shader='''
		#version 330

		in vec3 v_color;

		out vec3 f_color;

		void main() {
			f_color = v_color;
		}
	''',
)

vertexData = np.array([
	#  x, y, z, r, g, b
	1,-1,-1, np.random.rand(), np.random.rand(), np.random.rand(),
	1, 1,-1, np.random.rand(), np.random.rand(), np.random.rand(),
	-1, 1,-1, np.random.rand(), np.random.rand(), np.random.rand(),
	-1,-1,-1, np.random.rand(), np.random.rand(), np.random.rand(),
	1,-1, 1, np.random.rand(), np.random.rand(), np.random.rand(),
	1, 1, 1, np.random.rand(), np.random.rand(), np.random.rand(),
	-1, 1, 1, np.random.rand(), np.random.rand(), np.random.rand(),
	-1,-1, 1, np.random.rand(), np.random.rand(), np.random.rand()
])
indices = np.array([
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

vertexData.reshape(-1,6)

vbo = ctx.buffer(vertexData.astype('f4').tobytes())
ibo = ctx.buffer(indices.astype('i4').tobytes())
vao = ctx.vertex_array(prog, [
	(vbo, '3f 3f', 'pos','in_color'),
	], ibo)
#vao = ctx.simple_vertex_array(prog, vbo, 'pos', 'in_color', index_buffer=ibo)

# fbo = ctx.simple_framebuffer((512, 512))
# fbo.use()
# fbo.clear(0.0, 0.0, 0.0, 1.0)

# Uniforms.
## Model.
## View.
p = Quaternion(axis=[1, 0, 0], angle=-np.pi*0.5)
viewMat = p.transformation_matrix
## Projection.
projMat = mt.getProjection(60/180*np.pi,1,0.1,10)
print('proj:\n',projMat)
#projMat = np.eye(4)

# vao.render(moderngl.TRIANGLES)
# Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()
# quit()

time = 0
while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			quit()
	# Re calculate matrices.
	modelMat = mt.getScale(1)
	modelMat = Quaternion(axis=[1, 2, 3], angle=time * 0.02).transformation_matrix @ modelMat
	modelMat = mt.getTranslation(np.array([0,4,0])) @ modelMat

	# Send matrices.
	prog['model'].value = tuple(modelMat.T.flatten())
	prog['view'].value = tuple(viewMat.T.flatten())
	prog['proj'].value = tuple(projMat.T.flatten())
	# Render.
	ctx.clear(0.9, 0.9, 0.9, 1.)
	vao.render(moderngl.TRIANGLES)

	pygame.display.flip()
	time = time + 1
	pygame.time.wait(6)