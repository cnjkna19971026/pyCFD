import matplotlib.pyplot   as plt
import numpy        as np


x = np.point(-5 ,5 ,10)

y = x**2

plt.plot(x,y)

plt.xlabel('x')
plt.ylabel('y')

plt.title('graph of y = x^2')

plt.grid(True)

plt.show()

