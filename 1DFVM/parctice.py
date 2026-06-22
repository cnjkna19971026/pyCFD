
import numpy as np

# 1D heat transfer problem

# geo
L = 1.0, k = 10.0 
# boundary condition
T_left  = 200 , T_right = 250

# source
S = 1e3

# cell number 
N = 20
 
def solveFVM(N):
    
    dx = L/N
    
    xc = np.linspace(dx/2,L-dx/2,N)
    
    aE = aW = k/dx

    aP = aE + aW

    aB = k/dx/2

    dS = s*dx 
    
    A = np.zeros((N,N))
    b = np.zeros(N)

    for i in range(N):
        if i==0 :
            A[i,i]   = aE + aB
            A[i,i+1] = -aE
            b[i]   = dS + aB*T_left
        elif i == N-1 :
            A[i,i] = aW + aB
            A[i,i-1] = -aW
            b[i]     = dS + aB*T_right
        elif :
            A[i,i-1] = -aW
            A[i,i]     = aP
            A[i,i+1] = -aE
            b[i]       = dS
            
    T = np.linalg.solve(A,b)

    return xc, T



