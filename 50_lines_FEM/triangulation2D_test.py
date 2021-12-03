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

    #def define_boundaries(self):

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
        
        #print(self.Boundary)
        
        #self.entities_flags = {}
        #for flag, value in mesh.field_data.items():
        #    self.entities_flags[flag] = mesh.field_data[flag][0]
        #input_point_data = mesh.point_data
        #print("field_data:",mesh.field_data)
        #print(mesh.cell_data_dict)
        print(mesh.cell_sets)
        #print(mesh.get_cell_data('gmsh:physical', cell_type='line'))
        #stop
        #bound = {}
        #for i, set in mesh.cell_sets_dict.items():
        #    print("i:",i)
        #    print("set:",set)
        #    for en, edges in set.items():
        #        if en == 'line':
        #            bound[i] = edges
        #print("bound:", bound)
        #print(mesh.__init__)
        #print(mesh.int_data_to_sets)

        #stop
        #input_cell_data = mesh.cell_data
        #print(input_cell_data)
        #Boundary_flags = mesh.cell_data['gmsh:physical'][0:len(Boundary)]
        #self.Boundary = np.asarray([np.concatenate([Boundary[i], Boundary_flags[i]]) for i in range(len(Boundary))], dtype=np.int64)
    
    ''' NOT WORKING with boundary
    def savemesh(self, filename):
        
        # save mesh in class simple_triangulation to .msh file using meshio
        
        points = self.Coordinates.tolist()
        cells = [ ( "line", self.Boundary[0:2,:].tolist()), 
                    ("triangle", self.Elements.tolist() ) ]
        #cells_data = {"boundary": [np.ones(self.Boundary.shape[0]), 2*np.ones(self.Elements.shape[0])]}
        #point_data = {}
        cells = {"triangle": self.Elements.tolist() }
        #write_points_cells(filename,points, cells)
        mesh = meshio.Mesh(points,cells)
        
        #mesh.write(filename)
    '''