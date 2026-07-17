import numpy as np
from timecal import timeit

#u = np.array((0,1,2,3,4,5))
#
#print(u)
#print(len(u))
#print(u[1:])
#print(u[0:-1]) # u[0:-1] = u[0:5] 
#print(u[0:5])
#
L =2
t = 10
nx = 81# 81 
ny = 81# 81 
nt = 100
c = 1 
sigma = 0.2
dx = L/(nx-1) 
dy = L/(ny-1)
u = np.ones(nx)
print("123")

def func_1():
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
    
    u = np.full((ny,nx),3 , dtype = float)
    un = u.copy()
    
    u[int(0.5/dy):int(1/dy +1),int(0.5/dx):int(1/dx + 1)] = 1.5
    
    for n in range(1,nt+1):
        un = u.copy()
        row , col = u.shape
        for j in range (1,row):
            for i in range(1,col):
                u[j,i] = un[j,i] - (c*dt_x/dx*(un[j,i] - un[j,i-1])) - (c*dt_y/dy*(un[j,i]-un[j-1,i]))
            u[0, :] = 1
            u[-1, :] = 1
            u[:, 0] = 1
            u[:, -1] = 1

    return u


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
    
    u[int(0.5/dy):int(1/dy +1),int(0.5/dx):int(1/dx + 1)] = 2
    
    for n in range(nt+1):
        un = u.copy()
        u[1:,1:] = un[1:,1:] - (c*dt_x/dx*(un[1:,1:] - un[1:,0:-1])) - (c*dt_y/dy*(un[1:,1:]-un[0:-1,1:]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    return u

def myprint(string):
    print(f"{string}")
    return string
test = 

timeit(myprint,"hello")
timeit(myprint("hello"))
timeit(func_2)
timeit(func_1)
