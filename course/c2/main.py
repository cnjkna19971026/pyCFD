import numpy as np
import matplotlib.pyplot as plt

import time,sys
# normal para
nx = 41

L = 2;dx = L/(nx-1)

u = np.ones(nx)
u2 = np.zeros(nx)
un = np.zeros(nx)
un2 = np.zeros(nx)
# wave equ para
# nt = 2500000
c  = 1 

# initial condition
u2[int(.5/dx):int(1/dx + 1)] = 2
u[int(.5/dx):int(1/dx + 1)] = 2

# heat equ para for analytical
T   = np.ones(nx)
T_left  = 100
T_right = 100
T[0]  = T_left
T[nx-1] = T_right

Tana = lambda x : T_left + (T_right - T_left)*x/L + S/(2*k)*x*(L-x)  

def std_heat_equ(nx):

    S = 1e3;k = 15 ;rho = 2330 ; Cp = 750 
    L = 2;dx = L/(nx-1)
    T_left  = 100
    T_right = 100
    dt = 0.025 

    T   = np.ones(nx)
    #Tn  = np.ones(nx)
    T[0]  = T_left
    T[nx-1] = T_right
    niter = 25

    for n in range(niter):
        Tn = T.copy()
        for i in range(nx-2):
            T[i+1] =(T[i] + T[i+2])/2 + (S/(k*2))*(dx**2)
    return T 

def tran_heat_equ(nx):
    S = 1e3;k = 15 ;rho = 2330 ; Cp = 750 
    L = 2;dx = L/(nx-1)

    dt = 0.025 # 0.025

    T_left  = 100
    T_right = 100

    T   = np.zeros(nx)
    T[0]  = T_left
    T[nx-1] = T_right

    niter = 25000
    #snap = [T.copy()]
    for n in range(niter):
        Tn = T.copy()
        for i in range(nx-2):
            T[i+1] = Tn[i+1] + k*dt / (rho*Cp)/dx**2 * (Tn[i+2] - 2*Tn[i+1]+Tn[i]) + S*dt/(rho*Cp)
        #snap.append(T.copy())

    return T

def wave_equ(u,un):
    for n in range(nt):
        un = u.copy()
        for i in range( nx):
            u[i] = un[i] - c * dt/dx * (un[i] - un[i-1])
    return u

    #for i in range(nx):

# ====== plot ======
fig ,(ax1 ,ax2) = plt.subplots(1,2,figsize = (12,5))

ax1.plot(np.linspace(0,2,nx),std_heat_equ(nx),'-k',marker = 'o',ms = 4 , label =f" {nx} liter std std_heat_equ")
ax2.plot(np.linspace(0,2,nx),tran_heat_equ(nx),'-k',marker = 'o',ms = 4 , label = f" {nx} liter trans heat eque")

ax1.legend()
ax2.legend()
plt.show()
