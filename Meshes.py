from dolfin import *
from mshr import *
import matplotlib.pylab as plt

# --- Square mesh generator ---
def create_unite_square_mesh(N, unstructured=False, plots=False):
    if unstructured:
        domain = Rectangle(Point((0.0, 0.0)), Point((1.0, 1.0)))
        mesh = generate_mesh(domain, N)
        print("Unstructured mesh: ")
        print("-> hmax = ", mesh.hmax())
        print("-> hmin = ", mesh.hmin())
        print("-> Number of elements = ", mesh.num_cells())
        if plots:
            plot(mesh)
            plt.show()
    else:
        mesh = UnitSquareMesh(MPI.comm_self, N, N)
        print("Structured mesh: ")
        print("-> hmax = ", mesh.hmax())
        print("-> hmin = ", mesh.hmin())
        print("-> Number of elements = ", mesh.num_cells())
        if plots:
            plot(mesh)
            plt.show()

    return mesh

# --- Rectangle mesh generator ---
def create_rectangle_mesh(N, P1, P2, unstructured = False, plots = False):
    if unstructured:
        domain = Rectangle(P1, P2)
        mesh = generate_mesh(domain, N)
        print("Unstructured mesh: ")
        print("-> hmax = ", mesh.hmax())
        print("-> hmin = ", mesh.hmin())
        print("-> Number of elements = ", mesh.num_cells())
        if plots:
            plot(mesh)
            plt.show()
    else:
        mesh = RectangleMesh(P1, P2, int(abs(P2[0] - P1[0]))*N, int(abs(P2[1] - P1[1]))*N)
        print("Structured mesh: ")
        print("-> hmax = ", mesh.hmax())
        print("-> hmin = ", mesh.hmin())
        print("-> Number of elements = ", mesh.num_cells())
        if plots:
            plot(mesh)
            plt.show()
    return mesh