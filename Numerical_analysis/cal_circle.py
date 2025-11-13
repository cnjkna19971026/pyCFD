
import numpy as np
import matplotlib.pyplot as plt


# perpose to calculate the area of y = x^2 ,
# 

# initial parameter and defeat
N = 100
realAns = 1.0/3.0

x_init  = 0
x_end   = 1
x_width = abs(x_end-x_init) / N 

# use array to store the x location
rect_x_position = []
rect_x_height   = []
 

# print (abs(x_end-x_init), " ",x_width) 
area =0

# 

# calculate total area
for i in range (N):
    
    y = x_init**2
    sub_area = y*x_width
    area += sub_area
    
    rect_x_position.append(x_init)
    rect_x_height.append(y)

    x_init += x_width
    #print(x_init)

# add f in print () , then we can use {} to add variable in the " " 

error = realAns - area 

print(f"===========result & Info============")
print(f"result for  N : {N}")
print(f"numercial ans : {area}")
print(f"real ans : {realAns}")
print(f"error : {error}")

# visualization 

x_true = np.linspace(0 , 1 , 100)

y_true = x_true ** 2

plt.plot(x_true , y_true)


plt.bar(
    rect_x_position,
    rect_x_height,
    width = x_width, # The width of each bar
    alpha=0.4,# Make them semi-transparent
    color='gray',
    align='edge'# IMPORTANT: Make the bar start at 'x', not be centered on it
)

plt.show()


