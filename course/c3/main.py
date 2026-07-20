import numpy as np
import matplotlib.pyplot as plt

import time,sys


# nx =41




#def diffusion_equ(nx)
#    L   = 2 
#    dx  = 2
#    nt  = 20
#    nu  = 0.3
#    sigma = 0.2
#    dt = sigma * dx**2/ nu 





def wave_equ(nx, L = 2,nt = 10,nu=0.3,sigma = 0.5,u_left = 3.0,_u = 3,u_high = 2, x_start = 0.5,x_end = 1):
    dx = L/(nx-1)
    u = np.full(nx,_u)
    un = np.ones(nx)
    
    u[int(x_start/dx):int(x_end/dx)]= u_high

    # we changed dt from a const into sigma * dx
    # sigma = u*dt / dx <= sigma(max) 
    # sigma*dx = u*dt
    
    #dt = 0.025 # orig dt (as a const)
    sigma = 0.5
    # 
    dt = sigma*dx/np.max(np.abs(u)) 


    # non-linear c = un[i]
    for n in range(nt):
        un = u.copy()
        for i in range(1,nx):
            u[i] = un[i] - un[i]*(dt/dx)*(un[i]-un[i-1])

    # const c = 1
    #for n in range(nt):
    #    un = u.copy()
    #    for i in range(1,nx):
    #        u[i] = un[i] - c*(dt/dx)*(un[i]-un[i-1])

    x = np.linspace(0,2,nx)
    return x ,u

def diffusion_equ(nx):
    L = 2
    dx = L/(nx-1)
    nt = 10
    u = np.full(nx,3.0)
    un = np.full(nx,3.0)
    nu = 0.3 

    sigma = 0.2
    dt = sigma*(dx**2)/nu
    
    u[int(0.5/dx):int(1.0/dx)]=1
    
    for n in range (nt):
        un = u.copy()
        for i in range(1,nx-1):
            u[i] = un[i] + nu*dt/(dx**2)*(un[i+1]-2*un[i]+un[i-1])
    x = np.linspace(0,2,nx)

    return x , u 

def diffusion_equ_2(nx):
    L = 2
    dx = L/(nx-1)
    nt = 10
    u = np.full(nx,3.0)
    un = np.full(nx,3.0)
    nu = 0.3 

    sigma = 0.2
    dt = sigma*(dx**2)/nu
    
    u[int(0.5/dx):int(1.0/dx)]=1
    
    for n in range (nt):
        un = u.copy()
        u[1:-1] = un[1:-1] + nu*dt/(dx**2)*(un[2:]-2*un[1:-1]+un[0:-2])
    x = np.linspace(0,2,nx)

    return x , u 



nlist = np.array([41  , 81 ])
#idx  = 4
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,5))
figlist = [ax1,ax2,ax3,ax4]

for ax ,i in zip(figlist[0:2],nlist):
    x , u = diffusion_equ(i)
    ax.plot(x,u,'-k',marker='x',ms = 2.5,label=f"{i} liter wave_equ")
    ax.legend();ax.set_xlabel("x (m)");ax.set_ylabel("u (m/s)")


for ax ,i in zip(figlist[2:4],nlist):
    x , u = diffusion_equ_2(i)
    ax.plot(x,u,'-k',marker='x',ms = 2.5,label=f"{i} liter wave_equ")
    ax.legend();ax.set_xlabel("x (m)");ax.set_ylabel("u (m/s)")



#for idx , (ax,i) in enumerate(zip(figlist,nlist)): 
#    x , u = wave_equ(i)
#    x1 , u1 = diffusion_equ(i)
#    if idx  <= 2:
#        ax.plot(x,u,'-k',marker='x',ms = 2.5,label=f"{i} liter wave_equ")
#        ax.legend();ax.set_xlabel("x (m)");ax.set_ylabel("u (m/s)")
#    else:
#        ax.plot(x1,u1,'-k',marker='x',ms = 2.5,label=f"{i} liter wave_equ")
#        ax.legend();ax.set_xlabel("x (m)");ax.set_ylabel("u (m/s)")
        
   # ax2.plot(x,u),'-k',marker='x',ms = 2.5,label="wave_equ")
   # ax3.plot(x,u),'-k',marker='x',ms = 2.5,label="wave_equ")
   # ax4.plot(x,u),'-k',marker='x',ms = 2.5,label="wave_equ")

#ax1.legend();ax1.set_xlabel("x (m)");ax1.set_ylabel("u (m/s)")
#fig.legend();fig.set_xlabel("x (m)");fig.set_ylabel("u (m/s)")

plt.show()
