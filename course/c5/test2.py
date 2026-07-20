import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D 

nx = 31 #31
ny = 31 #31
nt = 10
nu = 0.05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.25
#dt = sigma * (dx **2 ) / nu
dt = sigma * dx * dy / nu

y = np.linspace(0, 2, ny)
x = np.linspace(0, 2, nx)

v = np.full(nx,1.0)

def func(nt):
    u = np.full((ny,nx),float(1))
    X, Y = np.meshgrid(x, y)
    u[int(0.5/dy):int(1.0/dy + 1),int(0.5/dx ):int(1.0/dx +1 )] = 2
    
    for n in range(nt +1):
        un =u.copy()
        u[1:-1, 1:-1] = (un[1:-1,1:-1] + nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + nu * dt / dy**2 *(un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        #u[1:-1,1:-1] = un[1:-1,1:-1] + ( nu*dt/(dx**2) * (un[1:-1,2:]) - 2 * un[1:-1,1:-1] + un[1:-1,0:-2]) + ( nu*dt/(dy**2) * (un[2:,1:-1]) - 2 * un[1:-1,1:-1] + un[0:-2,1:-1])
    
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    fig = pyplot.figure()
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=True)

    ax.set_zlim(1, 2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
    
    pyplot.show()
    return u

func(nt=60)

def diffusion_1(nt ):
    nx = 51 #31
    dx =float( 2 / (nx - 1))
    nu = 0.2
    sigma = 0.25
    dt = sigma * (dx **2) / nu
    x = np.linspace(0, 2, nx)
    v = np.full(nx,1.0)

    v[int(0.5/dx ):int(1./dx)] = 2

    for n in range(nt):
        vn =v.copy()
        v[1:-1] = vn[1:-1] + ( nu*dt/(dx**2) * (vn[2:] - 2 * vn[1:-1] + vn[0:-2])) 
    x = np.linspace(0, 2, nx)

    return x ,v

def diffusion_2(nt ):
    nx = 51 #31
    dx =float( 2 / (nx - 1))
    nu = 0.2
    sigma = 0.25
    dt = sigma * (dx **2) / nu
    x = np.linspace(0, 2, nx)
    v = np.full(nx,1.0)

    v[int(0.5/dx ):int(1./dx)] = 2

    for n in range(nt+1):
        vn =v.copy()
        v[1:-1] = vn[1:-1] + ( nu*dt/(dx**2) * (vn[2:] - 2 * vn[1:-1] + vn[0:-2])) 
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    x = np.linspace(0, 2, nx)

    return x ,v

def diffusion_3(nt ):
    nx = 51 #31
    dx =float( 2 / (nx - 1))
    nu = 0.2
    sigma = 0.25
    dt = sigma * (dx **2) / nu
    x = np.linspace(0, 2, nx)
    v = np.full(nx,1.0)

    v[int(0.5/dx ):int(1./dx)] = 2

    for n in range(nt+1):
        vn =v.copy()
        v[:] = vn[:] + ( nu*dt/(dx**2) * (vn[1:] - 2 * vn[:] + vn[:-1])) 


    x = np.linspace(0, 2, nx)

    return x ,v

