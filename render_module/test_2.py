import moderngl # For graphics.
import numpy as np # Vectorized math.
from pyquaternion import Quaternion # Rotation.
from PIL import Image # For image files.
import utilities.math_tools as mt # Custom math library.

ctx = moderngl.create_standalone_context()
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
            //gl_Position = view * model * localPos;
			//gl_Position.z = -gl_Position.z;

			if(isEq(proj[3][2], -0.22222222, 0.000001)){
				v_color.x = v_color.y = v_color.z = 0.5;
			}
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

fbo = ctx.simple_framebuffer((512, 512))
fbo.use()
fbo.clear(0.0, 0.0, 0.0, 1.0)

# Uniforms.
## Model.
modelMat = mt.getScale(1)
modelMat = Quaternion(axis=[1, 1, 1], angle=1).transformation_matrix @ modelMat
modelMat = mt.getTranslation(0,4,0) @ modelMat
## View.
p = Quaternion(axis=[1, 0, 0], angle=-np.pi*0.5)
viewMat = p.transformation_matrix
## Projection.
projMat = mt.getProjection(60/180*np.pi,1,0.1,10)
print('proj:\n',projMat)
#projMat = np.eye(4)
## Send matrices.
prog['model'].value = tuple(modelMat.T.flatten())
prog['view'].value = tuple(viewMat.T.flatten())
prog['proj'].value = tuple(projMat.T.flatten())

vao.render(moderngl.TRIANGLES)

Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()