import numpy as np
# when do I need use self ?
# consider the variable time priod

class structureGrid:

    def __init__(self, nx,ny,nz,length = (1.0,1.0,1.0),orig=(0.0,0.0,0.0)):
        self.shape = (nx,ny,nz)

        # GET FACTOR ALLWAYS USE [ ] , NO MATTER LIST OR TUPLE
        self.x = np.linspace(orig[0],orig[0]+length[0],nx)
        self.y = np.linspace(orig[1],orig[1]+length[1],ny)
        self.z = np.linspace(orig[2],orig[2]+length[2],nz)

        X,Y,Z = np.meshgrid(self.x,self.y,self.z,indexing="ij")
        self.coords = np.stack([X,Y,Z],axis = -1) 
        
        self.field = {}

    def add_scalarfield(self, name , init = 0.0): 
        self.field[name] = np.full((self.shape),init)
        return self.field[name]

    def add_vectorfield(self, name , init = 0.0 ,diff_x = 0.0 ,diff_y = 0.0,diff_z = 0.0 ): 
        u = self.coords[:,None,None,None] +(init + diff_x)
        v = self.coords[None,:,None,None] +(init + diff_y)
        w = self.coords[None,None,:,None] +(init + diff_z)

        field_u = np.broadcast_to(diff_x[None,None,:],(nz,ny,nx))
        field_v = np.broadcast_to(diff_y[None,:,None],(nz,ny,nx))
        field_u = np.broadcast_to(diff_w[:,None,None],(nz,ny,nx))
        self.field[name] = 

        return self.field[name]

    def __getattr__(self, name):
        """讓 grid.T 這種寫法直接取到場。"""
        fields = self.__dict__.get('field', {})
        if name in fields:
            return fields[name]
        raise AttributeError(f"'{type(self).__name__}' 沒有屬性或場 '{name}'")

mygrid = structureGrid(2,2,2)

T = mygrid.add_scalarfield('Temperature' , 300)

print(mygrid.Temperature)
print(mygrid.shape)
print(mygrid.coords[0,0,:])
print(T)
try:
    print(mygrid.pressure)
except AttributeError as e:
    print(f"error : {e}")
