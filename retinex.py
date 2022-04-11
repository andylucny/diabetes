import numpy as np
import cv2

def GaussianBlurViaFFT(src, sigma=25):
    assert(src.dtype == np.float64)
    ksize = np.round(6*sigma+1) 
    ksize += 1 if ksize % 2 == 0 else 0
    kernel1D = cv2.getGaussianKernel(ksize,sigma).astype(np.float64)
    kernel2D = kernel1D @ kernel1D.T
    dst = cv2.filter2D(src,-1,kernel2D,borderType=cv2.BORDER_REPLICATE)
    return dst

def singleScaleRetinex(img, sigma, a=10):
    retinex = np.log10(img) - np.log10(GaussianBlurViaFFT(img, sigma))
    return retinex/np.log10(a)
    
def isodataThreshold(src, weight=0.5):
    assert(src.dtype == np.uint8)
    histogram, _ = np.histogram(src,bins=256,range=(0,255)) 
    count = np.cumsum(histogram)
    bins = np.arange(len(histogram)).astype(np.float)
    subtotal = np.cumsum(histogram*bins)
    threshold = 0
    while True:
        if threshold < 0 or threshold >= 256:
            return 0
        if count[threshold] == 0:
            return threshold
        background = subtotal[threshold] / count[threshold]
        if count[255] == count[threshold]:
            return threshold
        foreground = (subtotal[255] - subtotal[threshold]) / (count[255] - count[threshold])
        new_threshold = int((1.0 - weight) * background + weight * foreground)
        if new_threshold == threshold:
            return threshold
        threshold = new_threshold

def retinexBinarization(src, sz = 25, a = 10.0, threshold = -1):
    scan = np.copy(src)
    mask = (scan != scan) # avoid NaN
    minimum = np.min(scan[np.logical_not(mask)])
    maximum = np.max(scan[np.logical_not(mask)])
    scan[mask] = minimum - max((maximum-minimum)/5,0.1)
    minimum = np.min(scan)
    scan -= minimum
    scan /= (maximum - minimum + 1e-5)
    scan += 1.0
    scan = scan.astype(np.float64)
    retinex = singleScaleRetinex(scan, sz, a)
    values = retinex.reshape(-1)
    lindex = len(values)//10
    rindex = len(values) - lindex
    lvalue = np.partition(values,lindex)[lindex]
    rvalue = np.partition(values,rindex)[rindex]
    retinex[retinex < lvalue] = lvalue
    retinex[retinex > rvalue] = rvalue
    gray = cv2.normalize(retinex, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if threshold < 0:
        threshold = isodataThreshold(gray)
    dst = np.logical_and((gray > threshold),np.logical_not(mask))
    return dst.astype(np.uint8)*255, gray, threshold

