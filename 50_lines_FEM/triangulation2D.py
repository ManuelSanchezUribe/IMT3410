import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio # pip install meshio
import pygmsh
from meshio import CellBlock, write_points_cells

class simple_triangulation:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key=='Coordinates':
                self.Coordinates = value
            if key=='Elements':
                self.Elements = value
            if key=='Boundary':
                self.Boundary =value
            if key=='filename':
                self.filename = value
                self.loadmesh()
    def add_boundaries(self, **kwargs):
        self.BC = {}
        for key, value in kwargs.items():
            self.BC[key] = value

    def plot_mesh(self):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 1, 1)
        x = self.Coordinates[:,0]
        y = self.Coordinates[:,1]
        ax.triplot(x, y, triangles=self.Elements)
        plt.show()
    
    def loadmesh(self):
        '''
        load mesh using meshio
        '''
        mesh = meshio.read(self.filename, file_format='gmsh')

        #print(mesh.cells.data)
        self.Coordinates = mesh.points
        self.Elements = mesh.cells_dict['triangle']
        self.Boundary = mesh.cells_dict['line']
        