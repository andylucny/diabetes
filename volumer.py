import os
import cv2
import numpy as np
from retinex import retinexBinarization
from curvature import getCurvature
from splitter import split
from annotator import save

imagesFolder = 'inputs'
subfolders = []
for folder in os.listdir(imagesFolder):
    path = os.path.join(imagesFolder, folder)
    if os.path.isdir(path):
        subfolders.append(path)

paths = []
for subfolder in subfolders:
    print('found',subfolder)
    for fname in os.listdir(subfolder):
        path = os.path.join(subfolder, fname)
        if fname.endswith('jpg') or fname.endswith('tiff'):
            print('-',fname)
            paths.append(path)  
            
sigma = 15#7
threshold = 30
median = 2#3#4 # *2 -1
curvaturityThreshold = 100

def updateSigma( *args ):
    global sigma
    sigma = args[0]

def updateThreshold( *args ):
    global threshold
    threshold = args[0]

def updateMedian( *args ):
    global median
    median = args[0]

def updateCurvaturityThreshold( *args ):
    global curvaturityThreshold
    curvaturityThreshold = args[0]

cv2.namedWindow("Retinex")
cv2.createTrackbar("sigma", "Retinex", sigma, 50, updateSigma)
cv2.createTrackbar("threshold", "Retinex", threshold, 255, updateThreshold)
cv2.createTrackbar("median", "Retinex", median, 10, updateMedian)
cv2.createTrackbar("100 x curvaturity", "Retinex", curvaturityThreshold, 500, updateCurvaturityThreshold)
            
def process(img0,splitting=False):
    img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    _, gray, _ = retinexBinarization(img, sz = sigma, threshold = threshold) 
    if median > 1:
        gray = cv2.medianBlur(gray,2*median-1)
    _, binary = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    candidates = []
    for contour in contours:
        if len(contour) > 250:
            if splitting:
                subcontours = split(contour,img.shape)
                for subcontour in subcontours:
                    curvaturity = np.std(getCurvature(subcontour,6))
                    if curvaturity * 100 <= curvaturityThreshold:
                        candidates.append(subcontour)
            else:
                curvaturity = np.std(getCurvature(contour,6))
                if curvaturity * 100 <= curvaturityThreshold:
                    candidates.append(contour)
                
    return gray, candidates

draw = True
origin = True
splitting = False
if len(paths) == 0:
    os.exit(0)
i = 0
parts = None
image = cv2.imread(paths[i])
print(paths[i])
j = 0   
while True:
    gray, contours = process(image,splitting=splitting)
    if j >= len(contours):
        j = 0
    if origin:
        disp = np.copy(image)
    else:
        disp = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    
    if draw:
        for k, subcontour in enumerate(contours):
            if k == j:
                color = (0,255,255)
                thickness = 2
            else:
                color = (0,255,0)
                thickness = 1
            for q in range(len(subcontour)-1):
                cv2.line(disp,tuple(subcontour[q][0]),tuple(subcontour[q+1][0]),color,thickness)
        
        txt = str(sigma)+' '+str(threshold)+' '+str(median)
        if len(contours) > 0:
            curvaturity = np.std(getCurvature(contours[j],6))
            area = cv2.contourArea(contours[j])
            perimeter = cv2.arcLength(contours[j], closed=True)
            circularity = (perimeter**2) / (4*np.pi*area)
            txt += ' '+str(area)+' '+str(circularity)+' '+str(curvaturity)
        cv2.putText(disp,txt,(16,32),0,1,(0,255,0),2)
        
    if parts is not None:
        for subcontour in parts:
            color = (0,0,255)
            thickness = 2
            for q in range(len(subcontour)-1):
                cv2.line(disp,tuple(subcontour[q][0]),tuple(subcontour[q+1][0]),color,thickness)
        
    cv2.imshow('Retinex',cv2.resize(disp,(disp.shape[1]//2,disp.shape[0]//2)))
    key = cv2.waitKeyEx(10)
    if key & 0xFF == 27:
        break
    elif key == 9: 
        origin = not origin
    elif key == ord(' '): 
        draw = not draw
    elif key == 2555904: # right arrow
        if j < len(contours)-1:
            j += 1
    elif key == 2424832: # left arrow
        if j > 0:
            j -= 1        
    elif key == 2621440: # dn arrow
        if i < len(paths)-1:
            i += 1
        else:
            i = 0
        parts = None
        image = cv2.imread(paths[i])
        print(paths[i])
        j = 0
    elif key == 2490368: # up arrow
        if i > 0:
            i -= 1
        else:
            i = len(paths)-1
        image = cv2.imread(paths[i]) 
        print(paths[i])
        j = 0        
    elif key == 2228224: # PgDn
        prefix0 = paths[i][:paths[i].find('\\',paths[i].find('\\')+1)]
        prefix = prefix0
        while prefix == prefix0:
            if i < len(paths)-1:
                i += 1
            else:
                i = 0
            parts = None
            prefix = paths[i][:paths[i].find('\\',paths[i].find('\\')+1)]
        image = cv2.imread(paths[i])
        print(paths[i])
        j = 0
    elif key == 2162688: # PgUp
        for k in range(2):
            prefix0 = paths[i][:paths[i].find('\\',paths[i].find('\\')+1)]
            prefix = prefix0
            while prefix == prefix0:
                if i > 0:
                    i -= 1
                else:
                    i = len(paths)-1
                prefix = paths[i][:paths[i].find('\\',paths[i].find('\\')+1)]
        if i < len(paths)-1:
            i += 1
        else:
            i = 0
        parts = None
        image = cv2.imread(paths[i])
        print(paths[i])
        j = 0
    elif key == ord('a'):
        if len(contours) > 0:
            parts = split(contours[j],image.shape)
            print('parts',len(parts))
    elif key == ord('S'):
        if len(contours) > 0:
            save('contour',[contours[j]],[0])
    elif key == ord('s'):
        if len(contours) > 0:
            save('annotation'+paths[i][5:-4]+'.txt',[contours[j]],[5],merge=True)
            print('contour saved into','annotation'+paths[i][5:-4]+'.txt')
    elif key == ord('c'):
        splitting = not splitting
    elif key != -1:
        print(key, key&0xFF)

cv2.destroyAllWindows()
