
import numpy as np
from pyquaternion import Quaternion # Rotation.

class MathTools:
	def __init__(self):
		pass
	@staticmethod
	def getProjection (fov, ratio, n, f):
		t = 1/np.tan(fov*0.5)
		return np.array([
			[0.5*t,           0,           0,  0],
			[    0, 0.5*ratio*t,           0,  0],
			[    0,           0, (f+n)/(n-f), -1],
			[	 0,           0, 2*f*n/(n-f),  0]
		]).T
	@staticmethod
	def getTranslation (p):
		return np.array([
			[   1,    0,    0, 0],
			[   0,    1,    0, 0],
			[   0,    0,    1, 0],
			[p[0], p[1], p[2], 1],
		]).T
	@staticmethod
	def getScale(scale):
		return np.array([
			[scale, 0, 0, 0],
			[0, scale, 0, 0],
			[0, 0, scale, 0],
			[0, 0, 0, 1]
		]).T
	@staticmethod
	def angleDifference(f0,u0,f1,u1):
		# Gives the angle difference between two orientations described as an initial forward and uppward vector (f0, u0), and a final forward and uppward vector (f1, u1).
		q = MathTools.lookAt(f0,u0)
		p = MathTools.lookAt(f1,u1)
		return np.abs((p*q.conjugate).angle)
	@staticmethod
	def angleDifference2(f0,u0,f1,u1):
		# Only works if angle is less or equal to 90 degrees.
		r0 = np.cross(f0,u0)
		r1 = np.cross(f1,u1)
		theta = np.linalg.norm(
			np.cross(r0,r1) +
			np.cross(f0,f1) +
			np.cross(u0,u1)
		)
		diff2 = np.arcsin(theta/2)
		return diff2
	@staticmethod
	def lookAt (forward, up):
		forward = forward/np.linalg.norm(forward)
		up = up/np.linalg.norm(up)
		right = np.cross(forward,up)
		up = np.cross(right,forward)
		m11 = right[0]
		m12 = forward[0]
		m13 = up[0]
		m21 = right[1]
		m22 = forward[1]
		m23 = up[1]
		m31 = right[2]
		m32 = forward[2]
		m33 = up[2]
		if (m22+m33 > -m22-m33): # W > X.
			if (m11+m33 > -m11-m33): # W > Y.
				if (m11+m22 > -m11-m22): # W > Z.
					w = np.sqrt(m11+m22+m33+1)/2
					x = (m23-m32)/4/w
					y = (m31-m13)/4/w
					z = (m12-m21)/4/w
					
				else: # Z > W.
					z = np.sqrt(-m11-m22+m33+1)/2
					w = (m12-m21)/4/z
					x = (m31+m13)/4/z
					y = (m23+m32)/4/z
			else: # Y > W.
				if (-m11+m22 > m11-m22): # Y > X.
					y = np.sqrt(-m11+m22-m33+1)/2
					w = (m31-m13)/4/y
					x = (m12+m21)/4/y
					z = (m23+m32)/4/y
				else: # X > Y.
					x = np.sqrt(m11-m22-m33+1)/2
					w = (m23-m32)/4/x
					y = (m12+m21)/4/x
					z = (m31+m13)/4/x
		else: # X > W.
			if (m11-m22 > -m11+m22): # X > Y.
				if (m11-m33 > -m11+m33): # X > Z.
					x = np.sqrt(m11-m22-m33+1)/2
					w = (m23-m32)/4/x
					y = (m12+m21)/4/x
					z = (m31+m13)/4/x
				else: # Z > X.
					z = np.sqrt(-m11-m22+m33+1)/2
					w = (m12-m21)/4/z
					x = (m31+m13)/4/z
					y = (m23+m32)/4/z
			else: # Y > X.
				if (m22-m33 > -m22+m33): # Y > Z.
					y = np.sqrt(-m11+m22-m33+1)/2
					w = (m31-m13)/4/y
					x = (m12+m21)/4/y
					z = (m23+m32)/4/y
				else: # Z > Y.
					z = np.sqrt(-m11-m22+m33+1)/2
					w = (m12-m21)/4/z
					x = (m31+m13)/4/z
					y = (m23+m32)/4/z
		return Quaternion(w,-x,-y,-z)
	@staticmethod
	def shortestArc(v1,v2):
		# https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
		# Quaternion q;
		# vector a = crossproduct(v1, v2);
		# q.xyz = a;
		# q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
		# a = np.cross(v1,v2)
		# q = Quaternion(
		# 	x=a[0],
		# 	y=a[1],
		# 	z=a[2],
		# 	w=np.sqrt(np.square(v1).sum() * np.square(v2).sum()) + v1@v2
		# )
		# return q.unit

		# p = r*q
		# p*q' = r

		cross = np.cross(v1,v2)
		if np.linalg.norm(cross) == 0:
			return Quaternion()
		q = MathTools.lookAt(v1,cross)
		p = MathTools.lookAt(v2,cross)
		return p*q.conjugate
	@staticmethod
	def angleBetweenVectors(u,v):
		magU = np.linalg.norm(u)
		if magU == 0:
			return 0
		magV = np.linalg.norm(v)
		if magV == 0:
			return 0
		unitU = u / magU
		unitV = v / magV
		cos = unitU @ unitV
		angle = np.arccos(cos)

		return angle
	@staticmethod
	def unitTest():
		# lookAt(...)
		q = MathTools.lookAt(np.array([1,2,3]),np.array([4,5,6]))
		if \
			not MathTools.isEqual(q.w, -0.41528255829284505) or \
			not MathTools.isEqual(q.x, -0.35130649231948197) or \
			not MathTools.isEqual(q.y, -0.7712338417692615) or \
			not MathTools.isEqual(q.z, -0.3306395417093324) \
			:
			raise Exception('Failed unit test.')
		
		'''# shortestArc(...)
		a = np.array([1,2,3])
		b = np.array([4,5,6])
		q = MathTools.shortestArc(a,b)
		if not MathTools.isEqualVec3(q.rotate(a), b):
			raise Exception('Failed unit test.')'''
		
		# Angle difference.
		f0 = np.array([0,1,0])
		u0 = np.array([0,0,1])
		f1 = np.array([0,1,0])
		u1 = np.array([0,0,1])
		diff = MathTools.angleDifference(f0,u0,f1,u1)
		diff2 = MathTools.angleDifference2(f0,u0,f1,u1)
		if not MathTools.isEqual(diff, 0):
			raise Exception('Failed unit test.')
		if not MathTools.isEqual(diff2, 0):
			raise Exception('Failed unit test.')
		#
		num = int(1e3)
		f0 = np.array([0,1,0])
		u0 = np.array([0,0,1])
		for i in range(num):
			rotation1 = Quaternion.random()
			f1 = rotation1.rotate(f0) * (np.random.random()+1)
			u1 = rotation1.rotate(u0) * (np.random.random()+1)
			rotation2 = Quaternion.random()
			f2 = rotation2.rotate(f1) * (np.random.random()+1)
			u2 = rotation2.rotate(u1) * (np.random.random()+1)
			diff = MathTools.angleDifference(f1,u1,f2,u2)
			if not MathTools.isEqual(diff,np.abs(rotation2.angle)):
				print('rotation1',rotation1)
				print('rotation2',rotation2)
				raise Exception('Failed unit test.')

		#
		f0 = np.array([0,1,0])
		u0 = np.array([0,0,1])
		ang = 1.123
		rotation = Quaternion(axis=[1,2,3],angle=ang)
		f1 = rotation.rotate(f0)
		u1 = rotation.rotate(u0)
		diff = MathTools.angleDifference(f0,u0,f1,u1)
		diff2 = MathTools.angleDifference2(f0,u0,f1,u1)
		if not MathTools.isEqual(diff,ang):
			raise Exception('Failed unit test.')
		if not MathTools.isEqual(diff2,ang):
			raise Exception('Failed unit test.')

		# Angle between vectors.
		u = np.array([1,2,3])
		ang = 1.23
		rotation = Quaternion(axis=[-1,2,-1],angle=ang) # Orthogonal axis to u.
		v = rotation.rotate(u)*4.321
		ang2 = MathTools.angleBetweenVectors(u,v)
		if not MathTools.isEqual(ang,ang2):
			raise Exception('Failed unit test.')

		print('Test successful')
	@staticmethod
	def isEqualVec3(u, v, margin=0.000001):
		return MathTools.isEqual(u[0],v[0],margin) and MathTools.isEqual(u[1],v[1],margin) and MathTools.isEqual(u[2],v[2],margin)
	@staticmethod
	def isEqualQuaternion(p,q,margin=0.000001):
		return MathTools.isEqual(p.w,q.w,margin) and MathTools.isEqual(p.x,q.x,margin) and MathTools.isEqual(p.y,q.y,margin) and MathTools.isEqual(p.z,q.z,margin)
	@staticmethod
	def isEqual(value1, value2, margin=0.000001):
		return np.abs(value1 - value2) < margin

#MathTools.unitTest()

