import numpy as np
import cv2

def getCurvature(vecContourPoints, step=1):

    vecCurvature = np.zeros((len(vecContourPoints)),np.float64)

    if len(vecContourPoints) < step:
        return vecCurvature

    for i in range(len(vecContourPoints)):

        pos = vecContourPoints[i][0]

        iminus = i-step
        iplus = i+step
        pminus = vecContourPoints[iminus + len(vecContourPoints) if iminus < 0 else iminus][0]
        pplus = vecContourPoints[iplus - len(vecContourPoints) if iplus >= len(vecContourPoints) else iplus][0]

        f1stDerivative = (pplus - pminus) / (2*step)
        f2ndDerivative = (pplus - 2*pos + pminus) / (step**2)

        divisor = f1stDerivative[0]**2 + f1stDerivative[1]**2
        curvature2D =  abs(f2ndDerivative[1]*f1stDerivative[0] - f2ndDerivative[0]*f1stDerivative[1]) / (divisor **1.5) if divisor > 10e-8 else 1e20 #inf
        vecCurvature[i] = curvature2D

    return vecCurvature

#import matplotlib.pyplot as plt
#vecContourPoints = np.array([[[0,0]],[[0,0.5]],[[0,1]],[[1,1]],[[1,0.5]],[[1,0]]], np.int32)
#step = 1
#vecCurvature = getCurvature(vecContourPoints, step)
#plt.plot(vecCurvature)
#plt.show()

