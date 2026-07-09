import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from timecal import timeit
from sympy import init_printing
#init_printing(use_latex=True)

x, nu, t = sp.symbols('x nu t')
phi =(sp.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) + sp.exp(-(x - 4 * t - 2 * sp.pi)**2 / (4 * nu * (t + 1))))

phiprime = phi.diff(x)

#u = -2 * nu / phi * phiprime + 4
u = -2 * nu * (phiprime/phi) + 4

print(u)

### add lambdify for easy orginaze 
from sympy.utilities.lambdify import lambdify

ufun = lambdify((t,x,nu),u)

#print(ufun(1,4,3))

def burgers_equ(nx=101,nt=100,nu=0.07,t=0.0):
    dx = 2*np.pi/(nx-1)
    dt = dx*nu

    x  = np.linspace(0,2*np.pi,nx)
    un = np.empty(nx)
    #t   = 0
    u  = np.asarray([ufun(t,x0,nu) for x0 in x])
    for n in range(nt):
        un = u.copy()
        for i in range(1,nx-1):
            u[i] = un[i] - un[i]*dt/dx*(un[i]-un[i-1]) + nu*dt/dx**2*(un[i+1]-2*un[i]+un[i-1])

        u[0] = un[i]-un[0]*dt/dx*(un[0]-un[-2]) + nu*dt/dx**2*(un[1]-2*un[0]+un[-2])
        u[-1] = u[0]
        
    return x,u ,nu , nt ,dt

x0,u0 ,nu,nt,dt= burgers_equ(nt = 10,nu=0.01)

u_ana = np.asarray([ufun(nt* dt,xi,nu) for xi in x0])

(x0,u0,nu,nt,dt),elapsed = timeit(burgers_equ,nt=5,nu=0.01)


print(u0)

plt.figure(figsize= (11,7), dpi = 100)
plt.plot(x0,u0,'-o',lw = 2,label = 'Computational')
plt.plot(x0,u_ana,'-r',lw = 2,label = 'Analytical')
plt.xlim([0,2*np.pi])
plt.ylim([0,10])
plt.legend()
plt.show()


