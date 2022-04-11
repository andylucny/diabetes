import os
import cv2
import numpy as np
from curvature import getCurvature
from warshall import warshall

def split(contour, shape, minArc = 250, maxEndsDistance = 120, curvatureStep = 6, curvatureLimit = 0.75, circularityLimit = 1.35, details = False):

    found = []

    curvature = getCurvature(contour, curvatureStep)
    quality = np.int32(curvature < curvatureLimit)
            
    subcontours = []
    l = 0
    while l < len(quality)-1:
    
        subcontour = []
        while l < len(quality)-1 and quality[l] > 0:
            subcontour.append(contour[l])
            l += 1
        
        if subcontour != []:
            subcontour = np.expand_dims(np.concatenate(subcontour, axis=0),axis=1)
        subcontours.append(subcontour)
        
        while l < len(quality)-1 and quality[l] == 0:
            l += 1
            
    if quality[len(quality)-1] > 0 and quality[0] > 0 and len(subcontours) > 1:
        subcontours[0] = np.concatenate([subcontours[len(subcontours)-1],subcontours[0]],axis=0)
        subcontours.pop()
        
    connected = np.zeros((len(subcontours),len(subcontours)),np.bool)
    
    for j in range(len(subcontours)):
        if len(subcontours[j]) > minArc:
            for k in range(len(subcontours)):
                if len(subcontours[k]) > minArc and j+1 != k and j+1-len(subcontours) != k:
                    distanceToClose = cv2.norm(subcontours[k][0]-subcontours[j][-1])
                    if distanceToClose < maxEndsDistance:
                         connected[j][k] = True

    #print(connected)
                         
    _, pieces = warshall(connected)
    
    #print(pieces)
    
    used = np.array([len(subcontours[j]) <= minArc for j in range(len(subcontours))],np.bool)
    candidates = []
    for j in range(len(subcontours)):
        if not used[j] and len(pieces[j,j]) > 0:
            candidate = np.concatenate([subcontours[p] for p in pieces[j,j]],axis=0)
            candidates.append(candidate)
            for p in pieces[j,j]:
                used[p] = True
            
    for j in range(len(candidates)):
        if cv2.norm(candidates[j][0]-candidates[j][-1]) < maxEndsDistance:
            area = cv2.contourArea(candidates[j])
            if area > 2*minArc:
                rect = cv2.boundingRect(candidates[j])
                _,_,w,h = rect
                if 3*w > 2*h and 3*h > 2*w:
                    perimeter = cv2.arcLength(candidates[j], closed=True)
                    circularity = (perimeter**2) / (4*np.pi*area)
                    curvature = getCurvature(candidates[j],6)
                    curvaturity = np.std(curvature)
                    if circularity < circularityLimit:
                        border = False
                        for point in candidates[j]:
                            x, y = point[0]
                            if x == 0 or y == 0 or x == shape[1]-1 or y == shape[0]-1:
                                border = True
                                break
                        if not border:
                            #print(i,j,"circularity =",circularity,"curvaturity =",curvaturity)
                            found.append(candidates[j])

    if details:
        return found, subcontours, curvature
    else:
        return found

def load(path):    
    with open(path+'.txt','r') as f:
        candidates = []
        categories = []
        lines = f.readlines()
        for line in lines:
            values = line.split()
            category = int(values[0])
            candidate = []
            for k in range(int(values[1])):
                point = [[int(values[2*k+2]),int(values[2*k+3])]]
                candidate.append(point)
            candidates.append(candidate)
            categories.append(category)

    print('loading',path,len(candidates),'candidates')
    return candidates, categories

if __name__ == "__main__":
    contours, _ = load('contour')
    contour = contours[0]
    contour = np.array(contour,np.int32)

    rows, cols = np.max(contour,axis=0)[0]
    disp = np.zeros((rows,cols,3),np.uint8)
    
    parts, subcontours, curvature = split(contour,disp.shape,details=True)
    
    #import matplotlib.pyplot as plt
    #plt.plot(curvature)
    #plt.show()
    
    #parts = [contour]
    #parts = subcontours

    colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,255,0),(255,0,255),(255,255,255)]
    for k, subcontour in enumerate(parts):
        color = colors[k % len(colors)]
        thickness = 2
        for q in range(len(subcontour)-1):
            cv2.line(disp,subcontour[q][0],subcontour[q+1][0],color,thickness)

    cv2.imshow('splitting',cv2.resize(disp,(disp.shape[1]//2,disp.shape[0]//2)))
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    