import numpy as np
from timecal import timeit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm


nx =int(81)
ny =int(81)
L = float(2.0)
t = float(10.0)
dx = L/(nx-1)
dy = L/(ny-1)
sigma = float(0.2)
dt_x = sigma * dx
dt_y = sigma * dy


x = np.linspace(0,L,nx)
y = np.linspace(0,L,ny)

X,Y = np.meshgrid(x,y)

u = np.ones((ny,nx))
un = u.copy()


u[int(0.5/dy ):int(1/dy + 1),int(0.5/dx):int(1/dx)] = 2



def func_1():
    L =2
    t = 80
    nx = 81
    ny = 81 
    nt = 100
    c = 1 
    sigma = 0.2
    dx = L/(nx-1) 
    dy = L/(ny-1)
    dt_x = sigma * dx 
    dt_y = sigma * dy 
    
    x = np.linspace(0,L , nx)
    y = np.linspace(0,L , ny)
    
    u = np.full((ny,nx),1 , dtype = float)
    un = u.copy()
    v = np.ones((ny,nx))
    vn = v.copy()
    
    u[int(0.5/dy):int(1/dy +1),int(0.5/dx):int(1/dx + 1)] = 2
    v[int(0.5/dy):int(1/dy +1),int(0.5/dx):int(1/dx + 1)] = 2
    
    for n in range(nt+1):
        un = u.copy()
        vn = v.copy()
        row , col = u.shape
        for j in range (1,row):
            for i in range(1,col):
                u[j,i] = un[j,i] - (un[j,i]*c*dt_x/dx*(un[j,i] - un[j,i-1])) - (vn[j,i]*c*dt_y/dy*(un[j,i]-un[j-1,i]))
                v[j,i] = vn[j,i] - (un[j,i]*c*dt_x/dx*(vn[j,i] - vn[j,i-1])) - (vn[j,i]*c*dt_y/dy*(vn[j,i]-vn[j-1,i]))

                u[0, :] = 1
                u[-1, :] = 1
                u[:, 0] = 1
                u[:, -1] = 1

                v[0, :] = 1
                v[-1, :] = 1
                v[:, 0] = 1
                v[:, -1] = 1

    return u , v


def func_2(): 
    L =2
    t = 10
    nx = 81
    ny = 81 
    nt = 100
    c = 1 
    sigma = 0.2
    dx = L/(nx-1) 
    dy = L/(ny-1)
    dt_x = sigma * dx 
    dt_y = sigma * dy 
    
    x = np.linspace(0,L , nx)
    y = np.linspace(0,L , ny)
    
    u = np.full((ny,nx),1, dtype =float)
    un = u.copy()

    v = np.ones((ny,nx))
    vn = v.copy()
    
    u[int(0.5/dy):int(1/dy +1),int(0.5/dx):int(1/dx + 1)] = 2
    v[int(0.5/dy):int(1/dy +1),int(0.5/dx):int(1/dx + 1)] = 2
    
    for n in range(nt+1):
        un = u.copy()
        vn = u.copy()
        u[1:,1:] = un[1:,1:] - (un[1:,1:]*c*dt_x/dx*(un[1:,1:] - un[1:,0:-1])) - (vn[1:,1:]*c*dt_y/dy*(un[1:,1:]-un[0:-1,1:]))
        v[1:,1:] = vn[1:,1:] - (un[1:,1:]*c*dt_x/dx*(vn[1:,1:] - vn[1:,0:-1])) - (vn[1:,1:]*c*dt_y/dy*(vn[1:,1:]-vn[0:-1,1:]))
                
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    return u , v

def func_3(): 
    L =2
    t = 10
    nx = 81
    ny = 81 
    nt = 100
    mu = 1 
    sigma = 0.2
    dx = L/(nx-1) 
    dy = L/(ny-1)
    dt_x = sigma * dx 
    dt_y = sigma * dy 
    
    x = np.linspace(0,L , nx)
    y = np.linspace(0,L , ny)
    
    u = np.full((ny,nx),300.0, dtype = float)
    un = u.copy()

    v = np.full((ny,nx), 300.0,dtype =  float)
    vn = v.copy()
    
    u[int(0.5/dy):int(1/dy +1),int(0.5/dx):int(1/dx + 1)] = 2
    v[int(0.5/dy):int(1/dy +1),int(0.5/dx):int(1/dx + 1)] = 2
    
    for n in range(nt+1):
        un = u.copy()
        vn = u.copy()
        u[1:,1:] = un[1:,1:] + (mu*dt_x/(dx**2)*(un[1:,2:]-2*un[1:,1:] + un[1:,0:-1])) + (mu*dt_y/(dy**2)*(un[1:,1:]-2*un[1:,1:]+un[0:-1,1:]))
        v[1:,1:] = vn[1:,1:] + (mu*dt_x/(dx**2)*(vn[1:,1:]-2*vn[1:,1:] + vn[1:,0:-1])) + (mu*dt_y/(dy**2)*(vn[1:,1:]-2*vn[1:,1:]+vn[0:-1,1:]))
                
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    return u , v
u1 , v1 = func_1()
u2 , v2 = func_2()
u3 , v3 = func_3()

timeit(func_2)
timeit(func_3)
timeit(func_1)

# incorrect form
#fig ,(ax1 , ax2)= plt.subplots(1,1,2 ,figsize = (12,5) , dpi =100)
#
#ax1.add_subplot(1,1,1,projection='3d')
#ax1.plot_surface(X, Y, u1[:], cmap=cm.viridis)
#
#ax2 = fig.add_subplot(1,1,2,projection='3d')
#surf2 = ax1.plot_surface(X, Y, u2[:], cmap=cm.viridis)
#

fig , (ax1 , ax2, ax3) = plt.subplots(1,3,figsize = (12,5) ,dpi =100 , subplot_kw ={'projection':'3d'})

surf1 = ax1.plot_surface(X,Y ,v1,cmap = cm.viridis,rstride=2, cstride=2)
surf2 = ax2.plot_surface(X,Y ,u2,cmap = cm.viridis)
surf3 = ax3.plot_surface(X,Y ,u3,cmap = cm.viridis)

plt.show()
