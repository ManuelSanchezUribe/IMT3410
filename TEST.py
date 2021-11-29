import numpy as np
from FEM_Poisson import Poisson_solver
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def main():
    #Test_simple()

    Test_simple2()


def Test_simple2():
    # Triangulation
    Coordinates = np.array([[0,0],[1,0],[0.5,0.5],[0,1],[1,1], [0.5,1]], dtype=np.float64)
    Elements = np.array([[0,1,2],[0,2,3],[1,4,2],[4,5,2],[5,3,2]], dtype=np.int64)
    Dirichlet = np.array([[3,0],[0,1],[1,4]],dtype=np.int64)
    Neumann = np.array([[4,5],[5,3]],dtype=np.int64)
    Th = mesh(Coordinates, Elements, Dirichlet, Neumann)
    # problem
    c = lambda x,y: 1.0+0*x+0*y
    f = lambda x,y: 0*x+0*y
    uD = lambda x,y: x+y
    g = lambda x,y: 1.0+0*x+0*y
    problemdata = Poisson_problem(c,f,uD,g)

    uh = Poisson_solver(Th, problemdata)

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
    Neumann = np.array([],dtype=np.int64)
    Dirichlet = np.array([[0,1],[1,3],[3,2],[2,0]],dtype=np.int64)
    Th = mesh(Coordinates, Elements, Dirichlet, Neumann)

    # problem
    c = lambda x,y: 1.0+0*x+0*y
    f = lambda x,y: 0*x+0*y
    uD = lambda x,y: x+y
    g = lambda x,y: 1.0+0*x+0*y
    problemdata = Poisson_problem(c,f,uD,g)

    uh = Poisson_solver(Th, problemdata)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    x = Th.Coordinates[:,0]
    y = Th.Coordinates[:,1]
    ax1.plot_trisurf(x, y, uh, triangles=Th.Elements, cmap=plt.cm.Spectral)
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.triplot(x, y, triangles=Th.Elements)
    plt.show()

# Classes
class mesh:
    def __init__(self, Coordinates, Elements, Dirichlet, Neumann):
        self.Coordinates = Coordinates
        self.Elements = Elements
        self.Dirichlet = Dirichlet
        self.Neumann = Neumann

class Poisson_problem:
    def __init__(self, c, f, uD, g):
        self.c = c
        self.f = f
        self.uD = uD
        self.g = g
if __name__ == "__main__":
    main()