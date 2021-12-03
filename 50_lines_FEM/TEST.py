import numpy as np
from FEM_Poisson import Poisson_solver
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from triangulation2D import simple_triangulation

def main():
    
    #Test_simple()
    #Test_simple2()
    Test_batman()


def Test_simple2():
    '''
    Test with Dirichlet and Neumann Boundary condition
    '''
    # Define Triangulation raw
    Coordinates = np.array([[0,0],[1,0],[0.5,0.5],[0,1],[1,1], [0.5,1]], dtype=np.float64)
    Elements = np.array([[0,1,2],[0,2,3],[1,4,2],[4,5,2],[5,3,2]], dtype=np.int64)
    Boundary = np.array([[3,0],[0,1],[1,4], [4,5],[5,3]], dtype=np.int64)

    # Create mesh class
    Th = simple_triangulation(Coordinates=Coordinates, Elements=Elements, Boundary=Boundary)
    
    # Define Dirichlet and Boundary Edges
    def Is_edge_on_Dirichlet(edge):
        point = [0.5*(edge[0,0]+ edge[1,0]), 0.5*(edge[0,1]+ edge[1,1])]
        tol = 1e-12
        if abs(point[0]- 0)<tol or abs(point[1]- 0)<tol or abs(point[0]-1.0)<tol:
            return True
        else:
            return False
    Dirichlet = np.where([Is_edge_on_Dirichlet(Th.Coordinates[Th.Boundary[j,0:2],0:2]) for j in range(Th.Boundary.shape[0])])[0]
    Neumann = np.setdiff1d(range(Th.Boundary.shape[0]), Dirichlet)
    Th.add_boundaries(Dirichlet=Dirichlet, Neumann=Neumann)
    
    # Define problem data
    c = lambda x,y: 1.0+0*x+0*y
    f = lambda x,y: 0*x+0*y
    uD = lambda x,y: x+y
    g = lambda x,y: 1.0+0*x+0*y
    problemdata = Poisson_problem(c,f,uD,g)
    
    # Solve
    uh = Poisson_solver(Th, problemdata)

    # Plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    x = Th.Coordinates[:,0]
    y = Th.Coordinates[:,1]
    ax1.plot_trisurf(x, y, uh, triangles=Th.Elements, cmap=plt.cm.Spectral)
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.triplot(x, y, triangles=Th.Elements)
    plt.show()

def Test_simple():
    # Triangulation
    Coordinates = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]], dtype=np.float64)
    Elements = np.array([[0,1,4],[2,0,4],[1,3,4],[3,2,4]], dtype=np.int64)
    Boundary = np.array([[0,1],[1,3],[3,2],[2,0]],dtype=np.int64)
    Th = simple_triangulation(Coordinates=Coordinates, Elements=Elements, Boundary=Boundary)
    # pure Dirichlet condition
    Neumann = np.array([],dtype=np.int64)
    Dirichlet = np.array([[0,1],[1,3],[3,2],[2,0]],dtype=np.int64)
    Th.add_boundaries(Dirichlet=Dirichlet, Neumann=Neumann)

    # problem data
    c = lambda x,y: 1.0+0*x+0*y
    f = lambda x,y: 0*x+0*y
    uD = lambda x,y: x+y
    g = lambda x,y: 1.0+0*x+0*y
    problemdata = Poisson_problem(c,f,uD,g)

    # Solve
    uh = Poisson_solver(Th, problemdata)

    # plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    x = Th.Coordinates[:,0]
    y = Th.Coordinates[:,1]
    ax1.plot_trisurf(x, y, uh, triangles=Th.Elements, cmap=plt.cm.Spectral)
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.triplot(x, y, triangles=Th.Elements)
    plt.show()

def Test_batman():
    '''
    load mesh from .msh file
    '''
    Th = simple_triangulation(filename='batman2.msh')
    Dirichlet = Th.Boundary
    Neumann = np.array([],dtype=np.int64)
    Th.add_boundaries(Dirichlet=Dirichlet, Neumann=Neumann)

    # problem data
    c = lambda x,y: 1.0+0*x+0*y
    f = lambda x,y: 0*x+0*y +np.cos(10*y)*np.cos(10*x)*np.exp(-(x**2+y**2)/100)
    uD = lambda x,y: 0*x+0*y
    g = lambda x,y: .0+0*x+0*y
    problemdata = Poisson_problem(c,f,uD,g)

    uh = Poisson_solver(Th, problemdata)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    #ax1 = fig.add_subplot(1, 2, 2)
    x = Th.Coordinates[:,0]
    y = Th.Coordinates[:,1]
    ax1.plot_trisurf(x, y, uh, triangles=Th.Elements, cmap=plt.cm.inferno)
    #ax1.tricontourf(x, y, uh, Th.Elements, 50, cmap = plt.cm.inferno, )
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.triplot(x, y, triangles=Th.Elements)
    plt.show()


class Poisson_problem:
    def __init__(self, c, f, uD, g):
        self.c = c
        self.f = f
        self.uD = uD
        self.g = g
if __name__ == "__main__":
    main()