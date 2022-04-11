import os
import cv2
import numpy as np
from retinex import retinexBinarization
from curvature import getCurvature
from splitter import split

def process(img0):
    img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    threshold = 30 #150 #250
    _, gray, _ = retinexBinarization(img, sz = 15, threshold=threshold) 
    gray = cv2.medianBlur(gray,5)
    _, binary = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
    #cv2.imwrite('gray.png',gray)
    #cv2.imwrite('binary.png',binary)

    contours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    depth = np.zeros(len(contours),int)
    for i in range(len(contours)):
        if hierarchy[i][3] != -1:
            depth[i] = depth[hierarchy[i][3]] + 1

    found = []
    for i in range(len(contours)):
        contour = contours[i]
        if len(contour) > 250: #depth[i] == 2 or depth[i] == 4:
            candidates = split(contour,img.shape)
            for j in range(len(candidates)):
                area = cv2.contourArea(candidates[j])
                if area > 30000: # area
                    #rect = cv2.boundingRect(candidates[j])
                    #_,_,w,h = rect
                    #if 3*w > 2*h and 3*h > 2*w: # rough shape
                    perimeter = cv2.arcLength(candidates[j], closed=True)
                    circularity = (perimeter**2) / (4*np.pi*area)
                    curvature = getCurvature(candidates[j],6)
                    curvaturity = np.std(curvature)
                    if circularity < 1.9: #1.35 for circles
                        border = False
                        for point in candidates[j]:
                            x, y = point[0]
                            if x == 0 or y == 0 or x == img.shape[1]-1 or y == img.shape[0]-1:
                                border = True
                                break
                        if not border:
                            if curvaturity < 0.2:
                                print(i,j,"circularity =",circularity,"curvaturity =",curvaturity)
                                found.append(candidates[j])
    
    sub = np.copy(img0)
    for subcontour in found:
        #color = (np.int(np.random.uniform(80,255)),np.int(np.random.uniform(80,255)),np.int(np.random.uniform(80,255)))
        color = (0,0,255)
        for q in range(len(subcontour)-1):
            cv2.line(sub,subcontour[q][0].astype(np.int),subcontour[q+1][0].astype(np.int),color,1)

    contains = np.zeros(len(found),np.int32)
    for i in range(len(found)):
        for j in range(len(found)):
            if i != j:
                point = found[j][0][0].astype(np.float32)
                #print(i,j,cv2.pointPolygonTest(found[i], point, False))
                if cv2.pointPolygonTest(found[i], point, False) > 0:
                    contains[i] += 1
    
    cells = []
    if len(contains) > 0:
        c0 = np.argmax(contains)
        c1 = np.argmin(contains)
        cell = [found[c0],found[c1]]
        if c1 == c0 or cv2.contourArea(cell[0]) < 1.1 * cv2.contourArea(cell[1]):
            cell[1] = None
        cells.append(cell)
        
    disp = np.copy(img0)
    for cell in cells:
        if cell[0] is not None:
            cv2.drawContours(disp,[cell[0]],0,(0,0,255),2)
        if cell[1] is not None:
            cv2.drawContours(disp,[cell[1]],0,(0,255,0),2)
    
    return disp, binary, sub, found

imagesFolder = 'inputs'
subfolders = []
for folder in os.listdir(imagesFolder):
    path = os.path.join(imagesFolder, folder)
    if os.path.isdir(path):
        subfolders.append(path)

for subfolder in subfolders:
    outfolder = 'out'+subfolder[2:]
    try:
        os.makedirs(outfolder)
    except:
        pass
    binfolder = 'binarie'+subfolder[5:]
    try:
        os.makedirs(binfolder)
    except:
        pass
    confolder = 'subcontour'+subfolder[5:]
    try:
        os.makedirs(confolder)
    except:
        pass
    annotfolder = 'annotation'+subfolder[5:]
    try:
        os.makedirs(annotfolder)
    except:
        pass
    print('processing',subfolder,'to',outfolder)
    for fname in os.listdir(subfolder):
        path = os.path.join(subfolder, fname)
        if fname.endswith('jpg') or fname.endswith('tiff'):
            print('-',fname)
            image = cv2.imread(path,cv2.IMREAD_COLOR)
            disp, binary, subcontours, candidates = process(image)
            cv2.imwrite('out'+path[2:],disp)
            cv2.imwrite('binarie'+path[5:],binary)
            cv2.imwrite('subcontour'+path[5:],subcontours)
            with open('annotation'+path[5:-4]+'.txt','w') as f:
                for candidate in candidates:
                    s = '0 ' + str(len(candidate))
                    for point in candidate:
                        s += '  ' + str(point[0][0]) + ' ' + str(point[0][1])
                    f.write(s+'\n')

