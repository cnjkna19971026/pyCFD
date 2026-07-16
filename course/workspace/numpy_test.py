import numpy as np

#nx,ny,nz = 3,3,3
#
#x = np.linspace(0,3,nx)
#y = np.linspace(0,3,ny)
#z = np.linspace(0,3,nz)
#
#X ,Y ,Z = np.meshgrid(x,y,z,indexing='ij')
#matrix_shpae = (nx,ny,nz)
#
#print(X)
#
#print("============================")
#
#T = np.full((nx,ny,nz),300)
#T[1,1,1] = 400 
#print(T.dtype)
#print("============================")
#T = np.full((nx,ny,nz),300,dtype = float)
#print(T.dtype)
#print("============================")
#coords = np.stack([X,Y,Z],axis=-1)
#
#
##print(coords[0,1,1])
#print(coords[1,1,1],T[1,1,1])
#

# {} has two data type, which is set & dict
#field = {"Velocity","gravity"} # this is set

# how to initial dict
field = {"velocity":None , "gravity":None}

# or 
# field = dict.fromkeys(["velocity","gravity"],0.0)

#Temp = [10,10,10]
# in dict the key word must be hashable (immutable)
# OK : int , float , string , tuple
# Not ok: adarray(numpy array) , dict , list 
field[ "temp" ] = [ 10,10,10]

print(field)
print(field["temp"])
