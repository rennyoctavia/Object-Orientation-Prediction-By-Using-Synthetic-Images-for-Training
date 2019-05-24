
import numpy as np
from pyquaternion import Quaternion # Rotation.
from .utilities.math_tools import MathTools # Custom math library.
import moderngl # For graphics.

class Mesh:
	def __init__(self):
		self.position = np.array([0,0,0]).astype(float)
		self.orientation = Quaternion(axis=[1, 0, 0], angle=0)
		self.size = np.ones(3).astype(float)
		self.vao = None
		self.shaderProgram = None
		self.texture = None
		self.specularMap = None
		self.dimmer = 1
	@staticmethod
	def getShaderProgram(ctx):
		return ctx.program(
			vertex_shader='''
				#version 330

				in vec3 pos;
				in vec2 texCoord;
				in vec3 normal;

				uniform mat4 model;
				uniform mat4 view;
				uniform mat4 proj;
				uniform vec3 size;

				out vec2 v_texCoord;
				out vec3 v_normal;
				out vec3 v_worldPosition;
				out mat4 v_model;

				bool isEq(float a, float b, float margin);

				void main() {
					vec4 localPos = vec4(pos * size, 1.0);
					vec4 worldPosition = model * localPos;
					gl_Position = proj * view * worldPosition;

					v_model = model;
					v_texCoord = texCoord;
					v_normal = mat3(model) * normal;
					v_worldPosition = vec3(worldPosition);
				}
				bool isEq (float a, float b, float margin) {
					return abs(a-b) < margin;
				}
			''',
			fragment_shader='''
				#version 330

				in vec2 v_texCoord;
				in vec3 v_normal;
				in vec3 v_worldPosition;
				in mat4 v_model;

				uniform vec3 camPos;
				uniform sampler2D texSampler;
				uniform sampler2D specSampler;
				uniform bool useSpec;
				uniform float dimmer;
				uniform vec3 lightColor;

				out vec4 f_color;
				
				vec3 calcColor (vec3 lightDir, vec3 viewDir, vec3 normal, vec4 color, float specularFactor);

				void main() {
					vec4 texCol = vec4(texture(texSampler, v_texCoord).rgb, 0.0);
					float specFac;
					if (useSpec){
						specFac = vec4(texture(specSampler, v_texCoord).rgb, 0.0).x * 20.;
					} else {
						specFac = 0.4;
					}
					vec3 lightDir = vec3(1., 1., -1.);
					vec3 viewDir = v_worldPosition - camPos;
					vec3 color = calcColor (
						normalize(lightDir),
						normalize(viewDir),
						normalize(v_normal + vec3(v_model*texCol-0.5)*.2),
						texCol,//vec4(1.,1.,1.,0.)
						specFac
					);
					color *= dimmer;
					color *= lightColor;
					f_color = vec4(color, 1.);
					//f_color = mix(f_color, vec4(0.0,0.,0.,1.), -pow(0.9,length(viewDir)) + 1);
				}
				vec3 calcColor (vec3 lightDir, vec3 viewDir, vec3 normal, vec4 color, float specularFactor) {
					// "lightDir", "viewDir" and "normal" is of unit length. "normal" is normal to the plane.
					float diffuseFactor = 1.;
					//float specularFactor = 1.;
					float ambientFactor = .4;

					// Diffuse shading.
					float diff = max(dot(normal, -lightDir), 0.0);
					// Specular shading.
					vec3 reflectDir = reflect(lightDir, normal); // reflect(I, N) = I - 2.0 * dot(N, I) * N.
					float spec = pow(max(dot(reflectDir, -viewDir), 0.0), 4.);

					// Self illuminant.
					diff += color.w;
					if (diff > 1.) {
						diff = 1.;
					}

					// Combine.
					vec3 diffuse  = diffuseFactor * diff * color.rgb;
					float smoothness = 0.; // 0 to infinity.
					float roughness = (color.r + color.g + color.b + smoothness) / (3. + smoothness);
					roughness -= 0.2;
					roughness = max(min(roughness,1.),0.);
					vec3 specular  = vec3(specularFactor * spec * roughness);
					vec3 ambient = ambientFactor * color.rgb;

					vec3 result = diffuse + specular + ambient;

					return result;
				}
			''',
		)
	def setShaderProgram(self,shaderProgram):
		self.shaderProgram = shaderProgram
	def loadModel(self, vertices, indices, texture, specularMap, ctx):
		if self.shaderProgram is None:
			raise Exception('Shader program is no yet set. \'setShaderProgram(shaderProgram)\' must be called first.')
		vbo = ctx.buffer(vertices.astype('f4').tobytes())
		ibo = ctx.buffer(indices.astype('i4').tobytes())
		self.vao = ctx.vertex_array(self.shaderProgram, [
			(vbo, '3f 2f 3f', 'pos','texCoord','normal'),
			], ibo)
		# Texture.
		self.texture = ctx.texture(texture.size, 3, texture.tobytes())
		self.texture.build_mipmaps()
		# Specular map.
		if specularMap is not None:
			self.specularMap = ctx.texture(specularMap.size, 3, specularMap.tobytes())
			self.specularMap.build_mipmaps()
		#
		self.shaderProgram['texSampler'].value = 0
	def render(self, cameraPosition, viewMat, projMat, shaderProgram, lightColor=np.array([1.,1.,1.])):
		if self.vao is None:
			raise Exception('Model not yet loaded. \'loadModel(...)\' must be called first.')
		if self.shaderProgram is None:
			raise Exception('Shader program is no yet set. \'setShaderProgram(shaderProgram)\' must be called first.')
		# Calculate model matrix.
		modelMat = self.orientation.transformation_matrix
		modelMat = MathTools.getTranslation (self.position) @ modelMat
		# Send matrices.
		self.shaderProgram['model'].value = tuple(modelMat.T.flatten())
		self.shaderProgram['view'].value = tuple(viewMat.T.flatten())
		self.shaderProgram['proj'].value = tuple(projMat.T.flatten())
		self.shaderProgram['camPos'].value = tuple(cameraPosition)
		self.shaderProgram['size'].value = tuple(self.size)
		#self.shaderProgram['dimmer'].value = tuple(self.dimmer)
		self.shaderProgram['dimmer'].value = self.dimmer
		self.shaderProgram['lightColor'].value = tuple(lightColor)
		# Textures.
		self.texture.use(0)
		self.shaderProgram['texSampler'].value = 0
		if self.specularMap is not None:
			self.specularMap.use(1)
			self.shaderProgram['useSpec'].value = True
			self.shaderProgram['specSampler'].value = 1
		else:
			self.shaderProgram['useSpec'].value = False
		# Render.
		self.vao.render(moderngl.TRIANGLES)




