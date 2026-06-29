import numpy as np
import matplotlib.pyplot as plt

import time,sys

from matplotlib.animation import FuncAnimation

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

def tran_heat_equ(nx):
    S = 1e2;k = 150 ;rho = 2330 ; Cp = 750 
    L = 2;dx = L/(nx-1)

    #alpha = k/(rho*Cp)
    #dt = 0.4*dx**2/alpha #11.65
    dt = 12

    T_left  = 100
    T_right = 100

    T   = np.zeros(nx)
    T[0]  = T_left
    T[nx-1] = T_right

    niter = 100
    snap = [T.copy()]
    for n in range(niter):
        Tn = T.copy()
        for i in range(nx-2):
            T[i+1] = Tn[i+1] + k*dt / (rho*Cp)/dx**2 * (Tn[i+2] - 2*Tn[i+1]+Tn[i]) + S*dt/(rho*Cp)
        snap.append(T.copy())

    return T , snap

    #for i in range(nx):

T , snap = tran_heat_equ(nx)

# ====== plot ======
fig ,(ax1 ,ax2) = plt.subplots(1,2,figsize = (12,5))
cmap =plt.cm.viridis
#ax1
# plot a line for each 10 step
for n in range(0,101,10) :
    ax1.plot(x,snap[n],color=cmap(n/100),lw = 1.5)

sm =plt.cm.ScalarMappable(cmap=cmap, norm = plt.Normalize(0,100))
fig.colorbar(sm,ax=ax1, label = "time step (nliter)")
ax1.set_xlabel("x (mm)") ; ax1.set_ylabel("T (C)")

#ax2
# plot a line for each 10 step
for n in range(0,101,10) :
    ax2.plot(x,snap[n],color=cmap(n/100),lw = 1.5)

sm =plt.cm.ScalarMappable(cmap=cmap, norm = plt.Normalize(0,100))
fig.colorbar(sm,ax=ax2, label = "time step (nliter)")
ax2.set_xlabel("x (mm)") ; ax2.set_ylabel("T (C)")

#ax1.legend()
#ax2.legend()
#plt.show()

# ========== animation

fig,ax=plt.subplots()
line,=ax.plot(x*1e3, snap[0], 'o-', ms=3)
ax.set_xlabel("x (mm)"); ax.set_ylabel("T (°C)")
ax.set_ylim(99.5, 102)                  # 固定 y 軸,否則每幀重縮放會看起來沒在動
title=ax.set_title("niter = 0")

def update(n):
    line.set_ydata(snap[n])
    title.set_text(f"niter = {n}")
    return line, title

anim=FuncAnimation(fig, update, frames=len(snap), interval=80, blit=False)
anim.save("heat_evo.gif", writer="pillow", fps=15)   # 存成 GIF


