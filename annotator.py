import os
import cv2
import numpy as np

imagesFolder = 'inputs'
subfolders = []
for folder in os.listdir(imagesFolder):
    path = os.path.join(imagesFolder, folder)
    if os.path.isdir(path):
        subfolders.append(path)

paths = [] 
for subfolder in subfolders:
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
    print('found',subfolder)
    for fname in os.listdir(subfolder):
        path = os.path.join(subfolder, fname)
        if fname.endswith('jpg') or fname.endswith('tiff'):
            print('-',fname)
            paths.append(path)
            
def load(path):    
    image = cv2.imread('subcontour'+path[5:],cv2.IMREAD_COLOR)
    with open('annotation'+path[5:-4]+'.txt','r') as f:
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
    return image, candidates, categories
    
def save(path,candidates,categories):
    with open('annotation'+path[5:-4]+'.txt','w') as f:
        for candidate, category in zip(candidates,categories):
            s = str(category) + ' ' + str(len(candidate))
            for point in candidate:
                s += '  ' + str(point[0][0]) + ' ' + str(point[0][1])
            f.write(s+'\n')

i = 0
if len(paths) == 0:
    os.exit(0)
    
image, candidates, categories = load(paths[i])
j = 0
for k in range(len(candidates)):
    if categories[k] > 0:
        j = k
        break
while True:
    disp = np.copy(image)
    if j < len(candidates):
        subcontour = candidates[j]
        subcategory = categories[j]
        color = (0,255,0)
        for q in range(len(subcontour)-1):
            cv2.line(disp,subcontour[q][0],subcontour[q+1][0],color,1)
        cv2.putText(disp,str(subcategory),(16,32),0,1,(0,255,0),2)
        
    cv2.imshow('annotation',disp)
    key = cv2.waitKeyEx()
    if key & 0xFF == 27:
        break
    elif key == 2555904: # right arrow
        if j < len(candidates)-1:
            j += 1
    elif key == 2424832: # left arrow
        if j > 0:
            j -= 1
    elif key == 2621440: # dn arrow
        if i < len(paths)-1:
            i += 1
        else:
            i = 0
        image, candidates, categories = load(paths[i])
        j = 0
        for k in range(len(candidates)):
            if categories[k] > 0:
                j = k
                break
    elif key == 2490368: # up arrow
        if i > 0:
            i -= 1
        else:
            i = len(paths)-1
        image, candidates, categories = load(paths[i])
        j = 0
        for k in range(len(candidates)):
            if categories[k] > 0:
                j = k
                break
    elif key == 2228224: # PgDn
        prefix0 = paths[i][:paths[i].find('\\',paths[i].find('\\')+1)]
        prefix = prefix0
        while prefix == prefix0:
            if i < len(paths)-1:
                i += 1
            else:
                i = 0
            prefix = paths[i][:paths[i].find('\\',paths[i].find('\\')+1)]
        image, candidates, categories = load(paths[i])
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
        image, candidates, categories = load(paths[i])
        j = 0
    elif key == ord('+'): # +
        if len(candidates) > 0:
            categories[j] += 1
            save(paths[i], candidates, categories)
    elif key == ord('-'): # -
        if len(candidates) > 0:
            if categories[j] > 0:
                categories[j] -= 1
                save(paths[i], candidates, categories)
    else:
        print(key, key&0xFF)

cv2.destroyAllWindows()
