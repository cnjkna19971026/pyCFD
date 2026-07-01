import numpy as np
import matplotlib.pyplot as plt

import time,sys


# nx =41
def wave_equ(nx):
    L = 2
    dx = L/(nx-1)
    nt = 25
    dt = 0.025
    u = np.ones(nx)
    c = 1
    #u = np.full(nx,4)
    un = np.ones(nx)
    
    u[int(0.5/dx):int(1/dx)]= 2.0

    # non-const c = un[i]
    #for n in range(nt):
    #    un = u.copy()
    #    for i in range(1,nx):
    #        u[i] = un[i] - un[i]*(dt/dx)*(un[i]-un[i-1])

    # const c = 1
    for n in range(nt):
        un = u.copy()
        for i in range(1,nx):
            u[i] = un[i] - c*(dt/dx)*(un[i]-un[i-1])

    x = np.linspace(0,2,nx)
    return x ,u

nlist = np.array([41 ,61 , 81 , 101])

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,5))
figlist = [ax1,ax2,ax3,ax4]

for ax,i in zip(figlist,nlist): 
    x , u = wave_equ(i)
    ax.plot(x,u,'-k',marker='x',ms = 2.5,label=f"{i} liter wave_equ")
    ax.legend();ax.set_xlabel("x (m)");ax.set_ylabel("u (m/s)")
   # ax2.plot(x,u),'-k',marker='x',ms = 2.5,label="wave_equ")
   # ax3.plot(x,u),'-k',marker='x',ms = 2.5,label="wave_equ")
   # ax4.plot(x,u),'-k',marker='x',ms = 2.5,label="wave_equ")

#ax1.legend();ax1.set_xlabel("x (m)");ax1.set_ylabel("u (m/s)")
#fig.legend();fig.set_xlabel("x (m)");fig.set_ylabel("u (m/s)")

plt.show()
