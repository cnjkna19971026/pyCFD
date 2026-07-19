import numpy as np

count = 0
nt = int(10)
c = int(1) 
u = np.linspace(0,int(2) ,int(10))  
v = np.linspace(0,int(2) ,int(10))  

def func_1():
    for i in range(nt+1):
        un = u.copy()
        u[1:-1] = un[1:-1] + c *(un[2:] + un[1:-1] + un[0])

    return u

def func_1_1():
    for i in range(1:nt):
        un = u.copy()
        u[1:-1] = un[1:-1] + c *(un[2:] + un[1:-1] + un[0])

    return u

def func_1_2():
    for i in range(1:nt+1):
        un = u.copy()
        u[1:-1] = un[1:-1] + c *(un[2:] + un[1:-1] + un[0])

    return u
print("=================================================")

def func_2():
    for i in range(nt+1):
        vn = v.copy()
        v[1:-1] = vn[1:-1] + c *(vn[2:] + vn[1:-1] + vn[0:-2])

    return u

u1 = func_1()
u2 = func_2()

print (u1)
print (u2)
print(u1==u2)
