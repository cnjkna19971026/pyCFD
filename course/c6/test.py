import numpy as np


nx = 11
dx = 2/(nx-1)
count = 0
nt = int(10)
c = int(1) 
x = np.linspace(0,2 ,nx)  
v = np.linspace(0,int(2) ,int(10))  
u = np.arange(1,12)

print(u)
print(f"len(x))     : ",len(u)      ,u)
print(f"len(x[1:-1]): ",len(u[1:-1]),u[1:-1])
print(f"len(x[:])   : ",len(u[:])   ,u[:])
print(f"len(x[0:])  : ",len(u[0:])  ,u[0:])
print(f"len(x[1:])  : ",len(u[1:])  ,u[1:])
print("=================================================")
print(x[:])


def func_1():
    for i in range (nt):
        print(x[:])
    return x

func_1()


# arr[0] = arr

#def func_1():
#    for i in range(nt+1):
#        un = u.copy()
#        u[1:-1] = un[1:-1] + c *(un[2:] + un[1:-1] + un[0])
#
#    return u
#
#def func_1_1():
#    for i in range(1:nt):
#        un = u.copy()
#        u[1:-1] = un[1:-1] + c *(un[2:] + un[1:-1] + un[0])
#
#    return u
#
#def func_1_2():
#    for i in range(1:nt+1):
#        un = u.copy()
#        u[1:-1] = un[1:-1] + c *(un[2:] + un[1:-1] + un[0])
#
#    return u
#print("=================================================")
#
#def func_2():
#    for i in range(nt+1):
#        vn = v.copy()
#        v[1:-1] = vn[1:-1] + c *(vn[2:] + vn[1:-1] + vn[0:-2])
#
#    return u

#u1 = func_1()
#u2 = func_2()
#
#print (u1)
#print (u2)
#print(u1==u2)
