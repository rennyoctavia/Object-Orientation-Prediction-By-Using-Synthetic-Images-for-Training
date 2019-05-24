
import numpy as np
import pickle
from utilities.math_tools import MathTools

pred = pickle.load(open('Octiba-Nima/render_module/prediction.pkl','rb'))
test = pickle.load(open('Octiba-Nima/render_module/y_test.pkl','rb'))

# Normalize preditions.
predN = (pred.reshape(-1,3)/np.sqrt(np.square(pred.reshape(-1,3)).sum(axis=1)).reshape(-1,1)).reshape(-1,6)

#

mrse = 0
totalAngleErr = 0
iterations = predN.shape[0]
for i in range(iterations):
	#
	p = pred[i]
	pN = predN[i]
	#p = p[]
	#
	t = test[i]
	squaredError = np.square(t-pN).sum()
	mrse += squaredError
	#
	angErr = MathTools.angleDifference(pN[0:3],pN[3:6],t[0:3],t[3:6])
	angErr2 = MathTools.angleDifference(p[0:3],p[3:6],t[0:3],t[3:6])
	totalAngleErr += angErr
mrse /= iterations
mrse = np.sqrt(mrse)
totalAngleErr /= iterations
print(mrse)
print(totalAngleErr*180/np.pi)







