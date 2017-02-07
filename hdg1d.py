// FEniCS/Dolfin code for solving the 1D Poisson equation
// with mixed boundary condition by the HDG method.
// Note: FEniCS-2016.2 is required to run this program. 
from dolfin import *
# Uniform mesh
NI = 4
mesh = UnitIntervalMesh(NI)
# Finite element spaces
W = FiniteElement("DG",interval, 1)
M = FiniteElement("CG",interval, 1)
V = FunctionSpace(mesh, W*M)
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
eta0 = 4
eta = eta0/h_avg
n = FacetNormal(mesh)
# Boundary Condition
bdry = FacetFunction("size_t", mesh,0)
class Right(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and \
    x[0] > 1 - DOLFIN_EPS
bdry.set_all(0)
Right().mark(bdry,1)
ds = Measure("ds",domain=mesh,\
     subdomain_data=bdry)
gN = Constant(-1) 
gD = Constant(0)  
bc = DirichletBC(V.sub(1),gD,'near(x[0],0)')
f = Expression("4",degree=0)
uex = Expression("x[0]*(3-2*x[0])",degree=2)
u,l = TrialFunctions(V)
v,m = TestFunctions(V)
# HDG scheme
a = dot(grad(u),grad(v))*dx\
 + dot(avg(grad(u)),jump(m-v,n))*dS\
 + inner(jump(grad(u),n),avg(m-v))*dS\
 + dot(avg(grad(v)),jump(l-u,n))*dS\
 + inner(jump(grad(v),n),avg(l-u))*dS\
 + eta*inner(avg(l-u),jump((m-v)*n,n))*dS\
 + eta*inner(jump((l-u)*n,n),avg(m-v))*dS\
 + eta0/h*(l-u)*(m-v)*ds\
 + inner(dot(grad(u),n),m-v)*ds\
 + inner(dot(grad(v),n),l-u)*ds
L = f*v*dx + gN*m*ds(1)
# Solve 
w = Function(V)
solve(a == L, w, bc)
# Display the values
u,l = w.split()
print(u.compute_vertex_values(mesh))
print(l.compute_vertex_values(mesh))
