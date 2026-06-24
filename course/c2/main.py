import numpy as np
import matplotlib.pyplot as plt

import time,sys

nx = 41
L = 2
dx = L/(nx-1)
nt = 25
dt = .025
c  = 1 

u = np.ones(nx)
u[int(.5/dx):int(1/dx + 1)] = 2
print(u)

plt.plot(np.linspace(0,2,nx),u,'-k',marker = 'o',ms = 4 , label = "wave eque")
plt.legend()
plt.show()
