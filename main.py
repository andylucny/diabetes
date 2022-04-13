import os
import cv2
import numpy as np
from retinex import retinexBinarization
from curvature import getCurvature
from splitter import split
from annotator import save

def process(img0):
    img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    threshold = 30 #150 #250
    _, gray, _ = retinexBinarization(img, sz = 15, threshold=threshold) 
    gray = cv2.medianBlur(gray,3) #5 #7
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
    areas = []
    for i in range(len(contours)):
        contour = contours[i]
        if len(contour) > 2*250: #depth[i] == 2 or depth[i] == 4:
            curvature = getCurvature(contour,6)
            curvaturity = np.std(curvature)
            if curvaturity < 1.0:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, closed=True)            
                circularity = (perimeter**2) / (4*np.pi*area)
                if circularity < 1.75: # speeding up
                    candidates = [contour]
                else:
                    candidates = split(contour,img.shape) # this is slow
                for j in range(len(candidates)):
                    area = cv2.contourArea(candidates[j])
                    if area > 30000: # area
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
                                if curvaturity < 0.75: # (median 3)  0.3 (median 5) #0.2 (median 7):
                                    print(i,j,"circularity =",circularity,"curvaturity =",curvaturity)
                                    found.append(candidates[j])
                                    areas.append(area)
    
    found = [ contour for area, contour in sorted(zip(areas,found), reverse=True) ]
    
    sub = np.copy(img0)
    for subcontour in found:
        #color = (np.int32(np.random.uniform(80,255)),np.int32(np.random.uniform(80,255)),np.int32(np.random.uniform(80,255)))
        color = (0,0,255)
        for q in range(len(subcontour)-1):
            cv2.line(sub,tuple(subcontour[q][0].astype(np.int32)),tuple(subcontour[q+1][0].astype(np.int32)),color,1)

    contains = np.zeros(len(found),np.int32)
    for i in range(len(found)):
        for j in range(len(found)):
            if i != j:
                point = tuple(found[j][0][0].astype(np.float32))
                #print(i,j,cv2.pointPolygonTest(found[i], point, False))
                if cv2.pointPolygonTest(found[i], point, False) > 0:
                    contains[j] += 1
    
    cells = []
    if len(contains) > 0:
        for c0 in range(len(contains)):
            if contains[c0] == 0:
                c1 = -1
                for c in range(len(found)):
                    point = tuple(found[c][0][0].astype(np.float32))
                    if cv2.pointPolygonTest(found[c0], point, False) > 0:
                        if c != c0 and cv2.contourArea(found[c0]) > 1.1 * cv2.contourArea(found[c]):
                            if c1 == -1 or contains[c] >= contains[c1]:
                                c1 = c
                cell = [c0,c1]
                cells.append(cell)
        
    disp = np.copy(img0)
    for cell in cells:
        colors = [(0,0,255),(0,255,0)]
        for c in range(2):
            if cell[c] >= 0:
                cv2.drawContours(disp,[found[cell[c]]],0,colors[c],2)
                
    categories = [ 0 for contour in found ]
    for cell in cells:
        for c in range(2):
            if cell[c] >= 0:
                categories[cell[c]] = c+1
    
    return disp, binary, sub, found, categories

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
            disp, binary, subcontours, candidates, categories = process(image)
            cv2.imwrite('out'+path[2:],disp)
            cv2.imwrite('binarie'+path[5:],binary)
            cv2.imwrite('subcontour'+path[5:],subcontours)
            save('annotation'+path[5:-4]+'.txt', candidates, categories, merge=True)
