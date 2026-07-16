import numpy as np
import matplotlib.pyplot as plt
import time,sys
from animation import snapshotrecorder , animation_snapshot, probRec , plt_proRec

from matplotlib.animation import FuncAnimation

dt = 10
nx =41
L = 2;dx = L/(nx-1)
x = np.linspace(0,L,nx)
# heat equ para for analytical
T   = np.ones(nx)
T_left  = 100
T_right = 100
T[0]  = T_left
T[nx-1] = T_right

Tana = lambda x : T_left + (T_right - T_left)*x/L + S/(2*k)*x*(L-x)  

def tran_heat_equ(nx, on_step = None ):
    S = 1e3;k = 150 ;rho = 2330 ; Cp = 750 
    L = 2;dx = L/(nx-1)

    #alpha = k/(rho*Cp)
    #dt = 0.4*dx**2/alpha #11.65
    dt = 10 #10

    T_left  = 100
    T_right = 100

    T   = np.zeros(nx)
    T[0]  = T_left
    T[nx-1] = T_right

    niter = 1000

    if on_step:
        on_step(0,T)

    for n in range(niter):
        Tn = T.copy()
        for i in range(nx-2):
            T[i+1] = Tn[i+1] + k*dt / (rho*Cp)/dx**2 * (Tn[i+2] - 2*Tn[i+1]+Tn[i]) + S*dt/(rho*Cp)
        if on_step:
            on_step(n+1,T)


    return T



rec = snapshotrecorder()
T  = tran_heat_equ(nx,on_step = rec )
animation_snapshot(x,rec.snap , filename = "heat_evo.gif")

tRec = probRec(i_probe = nx //2 , dt = 10)
T  = tran_heat_equ(nx,on_step = tRec)
plt_proRec(tRec.t_hist,tRec.T_hist)
