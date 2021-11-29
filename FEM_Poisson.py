import numpy as np

def Poisson_solver(mesh, problemdata):
    # Matrix assembling
    A, b = Matrix_assembling(mesh.Coordinates, mesh.Elements, mesh.Neumann, problemdata.c, problemdata.f, problemdata.g)
    # Dirichlet condition
    DirichletNodes = np.unique(mesh.Dirichlet)
    uh = np.zeros(mesh.Coordinates.shape[0], dtype=np.float64)
    for j in DirichletNodes:
        uh[j] = problemdata.uD(mesh.Coordinates[j,0], mesh.Coordinates[j,1])
    b += -A@uh
    Freenodes = np.setdiff1d(range(mesh.Coordinates.shape[0]), DirichletNodes)
    AFree = A[Freenodes,:][:,Freenodes]
    bFree = b[Freenodes]
    # Solve
    uh[Freenodes] = np.linalg.solve(AFree, bFree)
    return uh

def Matrix_assembling(Coordinates, Elements, Neumann, c, f, g):
    # Total number of vertices
    NN = Coordinates.shape[0]
    # Total number of elements
    NE = Elements.shape[0]
    # Total number of Neumann edges
    Nee = Neumann.shape[0]
    
    b = np.zeros(NN, dtype=np.float64)
    A = np.zeros((NN,NN), dtype=np.float64)
    for j in range(NE):
        # Coordinates of element j
        K = Coordinates[Elements[j,0:3],0:2]
        # Centroid
        xs = (K[0,0]+K[1,0]+K[2,0])/3.0
        ys = (K[0,1]+K[1,1]+K[2,1])/3.0
        detB_K = (K[1,0]-K[0,0])*( K[2,1]-K[0,1])-(K[1,1]-K[0,1])*(K[2,0]-K[0,0])
        measK = 0.5*abs(detB_K)
        # Local Stiffness Matrix
        G = np.linalg.solve(np.array([[1.0,1.0,1.0],[K[0,0],K[1,0],K[2,0]],[K[0,1],K[1,1],K[2,1]]], dtype=np.float64 ), np.array([[0,0],[1,0],[0,1]]))
        S = measK*c(xs,ys)*G@G.T
        for i in range(3):
            for k in range(3):
                A[Elements[j,i], Elements[j,k]] += S[i,k]
        # right-hand side
        bK = f(xs,ys)*measK/3.0
        b[Elements[j,0:3]] += bK
    # Neumann term
    for k in range(Nee):
        # Coordinates of Neumann edge k
        F = Coordinates[Neumann[k,0:2], 0:2]
        # midpoint
        xm = (F[0,0]+F[1,0])/2.0
        ym = (F[0,1]+F[1,1])/2.0
        measF = np.sqrt((F[1,0]-F[0,0])**2+(F[1,1]-F[0,1])**2) 
        bF = g(xm,ym)*measF/2.0
        b[Neumann[k,0:2]] += bF
    return A, b