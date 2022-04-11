import numpy as np

def warshall(mat):
    assert(mat.shape[0] == mat.shape[1])
    cnt = mat.shape[0]

    infinity = 1000000
    v = np.asarray(mat,np.int)
    v[v==0] = infinity

    w = np.zeros((cnt,cnt),np.object)
    for i in range(cnt):
        for j in range(cnt):
            w[i,j] = list(set([i,j]))
                
    for k in range(0,cnt):
        for i in range(cnt):
            for j in range(cnt):
                if v[i,k]+v[k,j] < v[i,j]:
                    v[i,j] = v[i,k]+v[k,j]
                    w[i,j] = list(set(w[i,k]+w[k,j]))

    return v, w

if __name__ == "__main__": 
    a = np.zeros((3,3),np.bool)
    a[2,1]=True
    a[1,0]=True
    a[1,1]=True
    a[0,2]=True
    v, w = warshall(a)
    
    