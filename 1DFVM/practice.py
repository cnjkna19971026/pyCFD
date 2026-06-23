
import numpy as np
import matplotlib.pyplot as plt

# 1D heat transfer problem


# ========== variable =========

# geo
L = 1.0  ; k = 10.0 # L(m) ; W/(m*k)
# boundary condition
T_left  = 300 ; T_right = 300
# source
S = 100
# cell number 
N = 20
 
def solveFVM(N):
    
    dx = L/N
    
    xc = np.linspace(dx/2,L-dx/2,N)
    
    aE = aW = k/dx

    aP = aE + aW

    aB = k/(dx/2)

    dS = S*dx 
    
    A = np.zeros((N,N))
    b = np.zeros(N)

    for i in range(N):
        if i==0 :
            A[i,i]      = aE + aB
            A[i,i+1]    = -aE
            b[i]        = dS + aB*T_left
        elif i == N-1 :
            A[i,i]      = aW + aB
            A[i,i-1]    = -aW
            b[i]        = dS + aB*T_right
        else :
            A[i,i-1]    = -aW
            A[i,i]      = aP
            A[i,i+1]    = -aE
            b[i]        = dS
            
    T = np.linalg.solve(A,b)

    return xc, T

# ============= analytical part ==============
# use more grid to represent continue
xact = np.linspace(0,L,400)

Tact = lambda x : T_left+(T_right-T_left)*(x/L)+(S/(2*k))*x*(L-x) 

xc, T = solveFVM(N)





# =========== plot fig ============

fig ,( ax1,ax2 ) =plt.subplots(1,2,figsize=(12,5))

# set figure inner color
# ax1.set_facecolor('lightblue')
# RGBA form
ax1.set_facecolor((0.5,0.5,0.9,0.1))

#set figure outter color
fig.patch.set_facecolor((0.5,0.5,0.5,0.3))

# ==============ax1================
ax1.plot(xc*1e3,T,'rx',ms=6,label='FVM(N=20)')
ax1.plot(xact*1e3,Tact(xact),'k-',label='Analytical')
ax1.set_xlabel('x[mm]');ax1.set_ylabel('T[K]'); ax1.set_ylim(290,310);ax1.grid(True); ax1.legend()

# =========== ax2 ===============
ax2.plot(xc*1e3,T,'rx',ms=6,label='FVM(N=20)')
ax2.plot(xact*1e3,Tact(xact),'-w',label='Analytical')
ax2.set_xlabel('x[mm]');ax2.set_ylabel('T[K]'); ax2.set_ylim(290,310);ax2.grid(True); ax2.legend()

plt.show()




