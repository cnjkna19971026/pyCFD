import numpy as np




vals = np.arange(1,4)
print(vals)
nx = 41
ny = 41
z0 = np.ones((3,3,3))
print(vals[:,None,None])
#myMtx_0 = vals[:,None,None]* z0

print(z0)
#print(myMtx_0)

myMtx = np.array((ny,nx))
print(myMtx)
print(myMtx.dtype)
print(myMtx.shape)

print(f"==================")
myMtx_1 = np.full((3,3,3),3)

print(myMtx_1)
print(myMtx_1.dtype)
print(myMtx_1.shape)

print(f"==================")

myMtx_2 = np.array([
                    [[1,1,1],[1,1,1],[1,1,1]],
                    [[2,2,2],[2,2,2],[2,2,2]], 
                    [[3,3,3],[3,3,3],[3,3,3]], 
                    ]
                  )
print(myMtx_2)
print(myMtx_2.dtype)
print(myMtx_2.shape)

