import numpy as np
import matplotlib.pyplot as plt

import time,sys



# normal para
nx = 41
L = 2
dx = L/(nx-1)

u = np.ones(nx)
u2 = np.zeros(nx)
un = np.zeros(nx)
un2 = np.zeros(nx)
# wave equ para
nt = 2500000
dt = .025
c  = 1 

# initial condition
u2[int(.5/dx):int(1/dx + 1)] = 2
u[int(.5/dx):int(1/dx + 1)] = 2
#print(u)


#print(un)

# heat equ para

T = np.ones(nx)
T2 = np.zeros(nx)
Tn = np.zeros(nx)
Tn2 = np.zeros(nx)

#nt = 25
#dt = .025
S = 1e2
k = 20
T[0]  = 100
T[nx-1] = 100
print(T)

def heat_equ(T,Tn):
    for n in range(nt):
        Tn = T.copy()
        for i in range(nx-2):
            T[i+1] =(T[i] + T[i+2])/2 + (S/(k*2))*(dx**2)
    return T 

#def heat_equ(T,Tn):
#    for n in range(nt):
#        Tn = T.copy()
#        for i in range(nx-2):
#            T[i] = 2*T[i+1]- T[i+2]+ (S/k)*(dx**2)
#    return T 

def wave_equ(u,un):
    for n in range(nt):
        un = u.copy()
        for i in range( nx):
            u[i] = un[i] - c * dt/dx * (un[i] - un[i-1])
    return u

    #for i in range(nx):

# ====== plot ======
fig ,(ax1 ,ax2) = plt.subplots(1,2,figsize = (12,5))

ax1.plot(np.linspace(0,2,nx),heat_equ(T,Tn),'-k',marker = 'o',ms = 4 , label = "un wave eque")
ax2.plot(np.linspace(0,2,nx),wave_equ(u2,un2),'-k',marker = 'o',ms = 4 , label = "un2 wave eque")

ax1.legend()
ax2.legend()
plt.show()
