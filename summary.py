import os
import cv2
import numpy as np

imagesFolder = 'outputs'
subfolders = []
for folder in os.listdir(imagesFolder):
    path = os.path.join(imagesFolder, folder)
    if os.path.isdir(path):
        subfolders.append(path)

for subfolder in subfolders:
    images = []
    print('found',subfolder)
    for fname in os.listdir(subfolder):
        path = os.path.join(subfolder, fname)
        if fname.endswith('jpg') or fname.endswith('tiff'):
            print('-',fname)
            image = cv2.imread(path)
            images.append(image)  
    
    if len(images) == 0:
        continue
        
    n = m = int(np.sqrt(len(images)))
    if n*m < len(images):
        n += 1
    if n*m < len(images):
        m += 1
        
    rows, cols = images[0].shape[:2]
    glob = np.zeros((m*rows, n*cols, 3),np.uint8)
    k = 0
    for i in range(m):
        for j in range(n):
            if k == len(images):
                break
            glob[i*rows:(i+1)*rows,j*cols:(j+1)*cols] = images[k]
            k += 1
        if k == len(images):
            break
    
    cv2.imwrite('summaries/'+subfolder[8:]+'.png',glob)
