from dolfin import *
import numpy as np
import scipy.sparse as sp
from datetime import date
import os
import math
import sys
import ipdb
import h5py
import pickle
import ipdb

'''
1. The fenics related data generator for model_new.py
2. Using sparse matrix linear algebra in Scipy
3. The new OP geometry is imported from Gmsh
'''

mesh = Mesh("./test_geo/test2.xml")
subdomains = MeshFunction("size_t", mesh, "./test_geo/test2_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "./test_geo/test2_facet_region.xml")
# def target_area(x):
#     # x[0] > 0.5 & x[0] < 2.0 & x[1] > 3.0 & x[1] < 4.5
#     return (x[0] <= 3.5) and (x[0] >= 0.5) and (x[1] <= 4.5) and (x[1] >= 2.0)
# def target2_area(x):
#     return (x[0] <= 5.5) and (x[0] >= 3.0) and (x[1] <= 2.0) and (x[1] >= 0.5)
# def target3_area(x):
#     return (x[0] <= 8.0) and (x[0] >= 6.0) and (x[1] <= 4.5) and (x[1] >= 3.0)

'''
Define more target area
'''
def target_area(x):
    return (x[0] <= 8.5) and (x[0] >= 6.5) and (x[1] <= 3.5) and (x[1] >= 1.0)
def target2_area(x):
    return (x[0] <= 9.0) and (x[0] >= 7.0) and (x[1] <= 3.5) and (x[1] >= 1.0)
def target3_area(x):
    return (x[0] <= 9.5) and (x[0] >= 7.5) and (x[1] <= 3.5) and (x[1] >= 1.0)

tq1_drt = './target_area/tq_point16.npy'
tq2_drt = './target_area/tq_point17.npy'
tq3_drt = './target_area/tq_point18.npy'

V0 = FunctionSpace(mesh,"DG",0)
V = VectorFunctionSpace(mesh,"CG",2)
P = FunctionSpace(mesh, "CG", 1)
W = V * P
T = FunctionSpace(mesh, "CG", 1)

# ke, velocity distinguishing coefficient 
ke = Function(V0)
ke_values = [0.05, 100]  # values of re in the two subdomains
for cell_no in range(len(subdomains.array())):
    if subdomains.array()[cell_no] == 34: # wall region
        ke.vector()[cell_no] = ke_values[1]
    else:
        ke.vector()[cell_no] = ke_values[0]
# ke.update()
# ke2, temperature distinguishing coefficient
ke2 = Function(V0)
ke2_values = [0.01*1.2, 1.0]
ke2_local_range = V0.dofmap().ownership_range()
n_ke2 = ke2_local_range[1] - ke2_local_range[0] # number of vector elements
for cell_no in range(len(subdomains.array())):
    if subdomains.array()[cell_no] == 34:
        ke2.vector()[cell_no] = ke2_values[1] # No.34 is wall's label
    else:
        ke2.vector()[cell_no] = ke2_values[0]
# ke2.update()

# import ipdb;ipdb.set_trace()
####
#### Stokes/ Navier-Stokes
####
re = Constant(0.05) # Renold number

v = TestFunction(V)
u = TrialFunction(V)
p = TrialFunction(P)
q = TestFunction(P)
te = TrialFunction(T)
de = TestFunction(T)

# Time-stepping
t = 10
dt = 10
k_dt = Constant(dt)
t_final = 300

str_test = 'x[0] >= 0.5 & x[0] <= 2.0 & x[1] >= 3.0 & x[1] <= 4.5 ? 1.0 : 0'
g = Expression( str_test) # heater position
g2 = Expression('x[0] >= 8.0 & x[0] <= 9.5 & x[1] >= 3.0 & x[1] <= 4.5 ? 1.0 : 0') # heater position 2

a1 = (1/k_dt)*inner(te,de)*dx + ke2*inner(grad(te),grad(de))*dx + inner(grad(te),u)*de*dx
# l1 = (1/k_dt)*inner(te0,de)*dx + g*de*dx
l1_express = (1/k_dt)*inner(te,de)*dx
l1_mat = assemble(l1_express)
l1 = l1_mat.array()

l2_express = ke2*inner(grad(te),grad(de))*dx
l2_mat = assemble(l2_express)
l2 = l2_mat.array()

g_express = g*de*dx
g_mat = assemble(g_express)
g_vector = g_mat.array()

g2_express = g2*de*dx
g2_mat = assemble(g2_express)
g2_vector = g2_mat.array()

n1_express = ke*inner(u,v)*dx
n1_mat = assemble(n1_express)
n1 = n1_mat.array()

n2_express = re*inner(grad(u),grad(v))*dx
n2_mat = assemble(n2_express)
n2 = n2_mat.array()

n3_express = inner(grad(p),v)*dx
n3_mat = assemble(n3_express)
n3 = n3_mat.array()

m_express = div(u)*q*dx
m_mat = assemble(m_express)
m = m_mat.array()

# Construct optimization variables
n_f = int(t_final/dt)
te_local_range = T.dofmap().ownership_range()
n_t = te_local_range[1] - te_local_range[0] # number of temperature elements
u_local_range = V.dofmap().ownership_range()
n_u = u_local_range[1] - u_local_range[0] # number of vector elements
p_local_range = P.dofmap().ownership_range()
n_p = p_local_range[1] - p_local_range[0] # number of temperature elements
# total number of variables is n_f*(n_t+1) + n_u + n_p + 2

# Fix DBC, temperature
tbc_zero = Constant(0)
tbc_express = tbc_zero*inner(te,de)*dx
tbc_mat = assemble(tbc_express)
bct1 = DirichletBC(T, tbc_zero, boundaries, 38)
bct2 = DirichletBC(T, tbc_zero, boundaries, 39)
bct3 = DirichletBC(T, tbc_zero, boundaries, 37)
bct4 = DirichletBC(T, tbc_zero, boundaries, 36)
bct = [bct1,bct2,bct3,bct4]
for bc in bct:
    bc.apply(tbc_mat) # apply Dirichlet boundary condition to temperature
tbc = tbc_mat.array()
tbc_point = []
for i in range(n_t):
    if tbc[i][i] > 0.5:
        tbc_point.append(i)
tbc_len = len(tbc_point)
tbc_point = np.asarray(tbc_point)
'''
# Dirichlet boundary controller 1, temperature
tbc_zero = Constant(0)
tbc_express = tbc_zero*inner(te,de)*dx
tbc_mat = assemble(tbc_express)
bct1 = DirichletBC(T, tbc_zero, boundaries, 36)
bct = [bct1]
for bc in bct:
    bc.apply(tbc_mat) # apply Dirichlet boundary condition to velocity
tbc = tbc_mat.array()
tbc_control = []
for i in range(n_t):
    if tbc[i][i] > 0.5:
        tbc_control.append(i)
tbc_control = np.array(tbc_control)
# Dirichlet boundary controller 2, temperature
tbc_zero = Constant(0)
tbc_express = tbc_zero*inner(te,de)*dx
tbc_mat = assemble(tbc_express)
bct1 = DirichletBC(T, tbc_zero, boundaries, 37)
bct = [bct1]
for bc in bct:
    bc.apply(tbc_mat) # apply Dirichlet boundary condition to velocity
tbc = tbc_mat.array()
tbc2_control = []
for i in range(n_t):
    if tbc[i][i] > 0.5:
        tbc2_control.append(i)
tbc2_control = np.array(tbc2_control)
# remove shared elements in dbc_point
tbc_point = [x for x in tbc_point if x not in tbc_control]
tbc_point = [x for x in tbc_point if x not in tbc2_control]
tbc_len = len(tbc_point)
tbc_point = np.asarray(tbc_point)
'''
# Fix Dirichlet Boundary condition, velocity
dbc_zero = Constant(0)
dbc_express = dbc_zero*inner(u,v)*dx
dbc_mat = assemble(dbc_express)
noslip = Constant((0,0))
bcu = DirichletBC(V, noslip, boundaries, 39) # define Dirichlet condition
bcu.apply(dbc_mat) # apply Dirichlet boundary condition to velocity
dbc = dbc_mat.array()
dbc_point = []
for i in range(n_u):
    if dbc[i][i] > 0.5:
        dbc_point.append(i)
dbc_len = len(dbc_point)
dbc_point = np.asarray(dbc_point)
# Dirichlet Boundary (inlet) points, normal
vbc_point = []
vbc_drct = Constant((1,0))
v_bc = DirichletBC(V, vbc_drct, boundaries, 36) # define Dirichlet condition
dbc_zero =  Constant(0)
dbc_express = dbc_zero*inner(u,v)*dx
dbc_mat = assemble(dbc_express)
dbc_zero_v = Constant((0,0))
dbc_express_rht = inner(dbc_zero_v,v)*dx
dbc_rht = assemble(dbc_express_rht)
v_bc.apply(dbc_mat,dbc_rht) # apply Dirichlet boundary condition to velocity
vbc_mat = dbc_mat.array()
vbc_r = dbc_rht.array()
for i in range(n_u):
    if (vbc_mat[i][i] > 0.5 and vbc_r[i] > 0.5):
        vbc_point.append(i)
vbc_len = len(vbc_point)
vbc_point = np.asarray(vbc_point)
# Dirichlet Boundary (inflow) points, para
vbc_point2 = []
vbc_drct = Constant((0,1))
v_bc = DirichletBC(V, vbc_drct, boundaries, 36) # define Dirichlet condition
dbc_zero =  Constant(0)
dbc_express = dbc_zero*inner(u,v)*dx
dbc_mat = assemble(dbc_express)
dbc_zero_v = Constant((0,0))
dbc_express_rht = inner(dbc_zero_v,v)*dx
dbc_rht = assemble(dbc_express_rht)
v_bc.apply(dbc_mat,dbc_rht) # apply Dirichlet boundary condition to velocity
vbc_mat = dbc_mat.array()
vbc_r = dbc_rht.array()
for i in range(n_u):
    if (vbc_mat[i][i] > 0.5 and vbc_r[i] > 0.5):
        vbc_point2.append(i)
vbc_len2 = len(vbc_point2)
vbc_point2 = np.asarray(vbc_point2)
# Dirichlet Boundary 2 (inlet 2) points, normal
vbc2_point = []
vbc_drct = Constant((1,0))
v_bc = DirichletBC(V, vbc_drct, boundaries, 37) # define Dirichlet condition
dbc_zero =  Constant(0)
dbc_express = dbc_zero*inner(u,v)*dx
dbc_mat = assemble(dbc_express)
dbc_zero_v = Constant((0,0))
dbc_express_rht = inner(dbc_zero_v,v)*dx
dbc_rht = assemble(dbc_express_rht)
v_bc.apply(dbc_mat,dbc_rht) # apply Dirichlet boundary condition to velocity
vbc_mat = dbc_mat.array()
vbc_r = dbc_rht.array()
for i in range(n_u):
    if (vbc_mat[i][i] > 0.5 and vbc_r[i] > 0.5):
        vbc2_point.append(i)
vbc2_len = len(vbc2_point)
vbc2_point = np.asarray(vbc2_point)
# Dirichlet Boundary 2 (inflow 2) points, para
vbc2_point2 = []
vbc_drct = Constant((0,1))
v_bc = DirichletBC(V, vbc_drct, boundaries, 37) # define Dirichlet condition
dbc_zero =  Constant(0)
dbc_express = dbc_zero*inner(u,v)*dx
dbc_mat = assemble(dbc_express)
dbc_zero_v = Constant((0,0))
dbc_express_rht = inner(dbc_zero_v,v)*dx
dbc_rht = assemble(dbc_express_rht)
v_bc.apply(dbc_mat,dbc_rht) # apply Dirichlet boundary condition to velocity
vbc_mat = dbc_mat.array()
vbc_r = dbc_rht.array()
for i in range(n_u):
    if (vbc_mat[i][i] > 0.5 and vbc_r[i] > 0.5):
        vbc2_point2.append(i)
vbc2_len2 = len(vbc2_point2)
vbc2_point2 = np.asarray(vbc2_point2)
# Dirichlet Boundary (outflow) points, orthogonal
vobc_point2 = []
vobc_drct = Constant((0,1))
vo_bc = DirichletBC(V, vobc_drct, boundaries, 38) # define Dirichlet condition
dbc_zero =  Constant(0)
dbc_express = dbc_zero*inner(u,v)*dx
dbc_mat = assemble(dbc_express)
dbc_zero_v = Constant((0,0))
dbc_express_rht = inner(dbc_zero_v,v)*dx
dbc_rht = assemble(dbc_express_rht)
vo_bc.apply(dbc_mat,dbc_rht) # apply Dirichlet boundary condition to velocity
vobc_mat = dbc_mat.array()
vobc_r = dbc_rht.array()
for i in range(n_u):
    if (vobc_mat[i][i] > 0.5 and vobc_r[i] > 0.5):
        vobc_point2.append(i)
vobc_len2 = len(vobc_point2)
vobc_point2 = np.asarray(vobc_point2)
######
### remove shared elements in dbc_point
######
dbc_point = [x for x in dbc_point if x not in vbc_point]
dbc_point = [x for x in dbc_point if x not in vbc_point2]
dbc_point = [x for x in dbc_point if x not in vobc_point2]
dbc_point = [x for x in dbc_point if x not in vbc2_point]
dbc_point = [x for x in dbc_point if x not in vbc2_point2]
dbc_len = len(dbc_point)
dbc_point = np.asarray(dbc_point)
# Fixed pressure in outflow area
pbc_point = []
pbc_drct = Constant(0)
p_bc = DirichletBC(P, pbc_drct, boundaries, 38) # define Dirichlet condition
pbc_zero =  Constant(0)
pbc_express = pbc_zero*inner(p,q)*dx
pbc_mat = assemble(pbc_express)
p_bc.apply(pbc_mat) # apply Dirichlet boundary condition to velocity
pbc_mat = pbc_mat.array()
for i in range(n_p):
    if (pbc_mat[i][i] > 0.5):
        pbc_point.append(i)
pbc_len = len(pbc_point)
pbc_point = np.asarray(pbc_point)
# Mark the target area
tq_point = []
tq_drct = Constant(0)
tq_bc = DirichletBC(T, tq_drct, target_area) # define Dirichlet condition
tq_zero = Constant(0)
tq_express = tq_zero*inner(te,de)*dx
tq_mat = assemble(tq_express)
tq_bc.apply(tq_mat) # apply Dirichlet boundary condition to velocity
tq_mat = tq_mat.array()
for i in range(n_t):
    if (tq_mat[i][i] > 0.5):
        tq_point.append(i)
tq_len = len(tq_point)
tq_point = np.asarray(tq_point)
# Mark the target area 2
tq_point2 = []
tq_drct = Constant(0)
tq_bc = DirichletBC(T, tq_drct, target2_area) # define Dirichlet condition
tq_zero = Constant(0)
tq_express = tq_zero*inner(te,de)*dx
tq_mat = assemble(tq_express)
tq_bc.apply(tq_mat) # apply Dirichlet boundary condition to velocity
tq_mat = tq_mat.array()
for i in range(n_t):
    if (tq_mat[i][i] > 0.5):
        tq_point2.append(i)
tq_len2 = len(tq_point2)
tq_point2 = np.asarray(tq_point2)
# Mark the target area 3
tq_point3 = []
tq_drct = Constant(0)
tq_bc = DirichletBC(T, tq_drct, target3_area) # define Dirichlet condition
tq_zero = Constant(0)
tq_express = tq_zero*inner(te,de)*dx
tq_mat = assemble(tq_express)
tq_bc.apply(tq_mat) # apply Dirichlet boundary condition to velocity
tq_mat = tq_mat.array()
for i in range(n_t):
    if (tq_mat[i][i] > 0.5):
        tq_point3.append(i)
tq_len3 = len(tq_point3)
tq_point3 = np.asarray(tq_point3)
# t1 = time.time()

np.save( tq1_drt, tq_point )
np.save( tq2_drt, tq_point2 )
np.save( tq3_drt, tq_point3 )
ipdb.set_trace()

t_range = range(n_t)
for i in tbc_point:
    t_range.remove(i)
# for i in tbc_control:
#     t_range.remove(i)
# for i in tbc2_control:
#     t_range.remove(i)
n_e1 = len(t_range)
v_range = range(n_u)
for i in dbc_point:
    v_range.remove(i)
for i in vbc_point:
    v_range.remove(i)
for i in vbc_point2:
    v_range.remove(i)
for i in vbc2_point:
    v_range.remove(i)
for i in vbc2_point2:
    v_range.remove(i)
for i in vobc_point2:
    v_range.remove(i)
n_e2 = len(v_range)
p_range = range(n_p)
for i in pbc_point:
    p_range.remove(i)
n_e3 = len(p_range)
'''
#########
# python filter is much slower than expected 
t2 = time.time()
t_range = filter((lambda x: sum(tbc_point==x)==0), range(n_t))
v_range = filter((lambda x: sum(dbc_point==x)==0), range(n_u))
v_range = filter((lambda x: sum(vbc_point==x)==0), v_range)
v_range = filter((lambda x: sum(vbc_point2==x)==0), v_range)
v_range = filter((lambda x: sum(vobc_point2==x)==0), v_range)
p_range = filter((lambda x: sum(pbc_point==x)==0), range(n_p))
n_e1 = len(t_range)
n_e2 = len(v_range)
n_e3 = len(p_range)
t3 = time.time()
print "t2 - t1: " + str(t2-t1) + '\n'
print "t3 - t2: " + str(t3-t2) + '\n'
import ipdb; ipdb.set_trace()
'''

import time
'''
t1 = time.time()
nuu = [] # sparse matrix list for Navier-Stokes, nuu[i][j,k]: i is for w, j is for v, k is for u
w = Function(V)
for i in range(0,n_u):
    coeff = np.zeros(n_u)
    coeff[i] = 1.0
    w.vector().set_local(coeff)
#    w.update()
    nuu_express = inner(grad(u)*v,w)*dx 
    nuu_mat = assemble(nuu_express)
    nuu.append(sp.csr_matrix(nuu_mat.array()))
lut = [] # sparse matrix list for heat trans, lut[i][j,k], i for te, j for de, k for v
w = Function(T)
for i in range(n_t):
    coeff = np.zeros(n_t)
    coeff[i] = 1.0
    w.vector().set_local(coeff)
#    w.update
    lut_express = inner(grad(w),u)*de*dx
    lut_mat = assemble(lut_express)
    lut.append(sp.csr_matrix(lut_mat.array()))
# import ipdb; ipdb.set_trace()
'''
############
# Test ipython parallel below
t2 = time.time()
from IPython import parallel
rc = parallel.Client()
lview = rc.load_balanced_view()
def lut_assemble(i=0, n_t=1):
    tmp_i = i
    from dolfin import *
    import numpy as np
    import scipy.sparse as sp
    mesh = Mesh("./test_geo/test2.xml")
    T = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh,"CG",2)
    u = TrialFunction(V)
    de = TestFunction(T)    
    coeff = np.zeros(n_t)
    coeff[tmp_i] = 1.0
    w = Function(T)
    w.vector().set_local(coeff)
    # w.update
    lut_express = inner(grad(w),u)*de*dx
    lut_mat = assemble(lut_express)
    return sp.csr_matrix(lut_mat.array())

lut_para = []
# import ipdb;ipdb.set_trace()
# ar = lut_assemble(0,n_t)
for i in range(n_t):
    ar = lview.apply(lut_assemble, i, n_t)
    lut_para.append(ar)
rc.wait(lut_para)
def nuu_assemble(i=0, n_u=1):
    tmp_i = i
    from dolfin import *
    import numpy as np
    import scipy.sparse as sp
    mesh = Mesh("./test_geo/test2.xml")
    V = VectorFunctionSpace(mesh,"CG",2)
    u = TrialFunction(V)
    v = TestFunction(V)    
    coeff = np.zeros(n_u)
    coeff[tmp_i] = 1.0
    w = Function(V)
    w.vector().set_local(coeff)
    # w.update
    nuu_express = inner(grad(u)*v,w)*dx
    nuu_mat = assemble(nuu_express)
    return sp.csr_matrix(nuu_mat.array())
nuu_para = []
for i in range(n_u):
    ar = lview.apply(nuu_assemble, i, n_u)
    nuu_para.append(ar)
rc.wait(nuu_para)
lut = []
nuu = []
for ele in lut_para:
    lut.append(ele.get())
for ele in nuu_para:
    nuu.append(ele.get())    
t3 = time.time()

'''
print "non parall: " + str(t2-t1) + '\n'
print "parall: " + str(t3-t2) + '\n'
# import ipdb; ipdb.set_trace()
print "Testing nonparall and parall results, lut:" + '\n'
for i in range(n_t):
    if ((lut[i] - lut_para[i].get()).min() != 0.0) and ((lut[i] - lut_para[i].get()).max() != 0.0):
        print "Not the same: " + i + ". \n"
print "Testing nonparall and parall results, lut:" + '\n'
for i in range(n_u):
    if ((nuu[i] - nuu_para[i].get()).min() != 0.0) and ((nuu[i] - nuu_para[i].get()).max() != 0.0):
        print "Not the same: " + i + ". \n"
'''
# import ipdb;ipdb.set_trace()

'''
#############
# Test 3D tensors
w = Function(V)
lut_dense = np.zeros((n_u,n_t,n_t))
for i in range(n_u):
    coeff = np.zeros(n_u)
    coeff[i] = 1
    w.vector().set_local(coeff)
    w.update()
    lut_express = inner(grad(te),w)*de*dx # lut[i,j,k]: i is for w, j is for de, k is for te
    lut_mat = assemble(lut_express)
    lut_temp = lut_mat.array()
    # Transpose lut,nuu to match row and col
    lut_dense[i,:,:] = lut_temp
for i in range(n_t):
    if ((lut[i] - lut_dense[:,:,i].T).min() != 0.0) and ((lut[i] - lut_dense[:,:,i].T).max() != 0.0):
        print "Not the same: " + i + ". \n"
'''

## get all coefficients
num_t = len(t_range) # number of temperature variables in OP
num_u = len(v_range) # number of velocity
num_p = len(p_range) # number of pressure
n_total = n_f*(num_t+1+1) + num_u + num_p + (1 + 1)*2 # 2 heaters and 2 fans
n_constraint = n_f*n_e1 + n_e2 + n_e3

def store_data( filename, filename_lut, filename_nuu,
                n_f, n_t, n_u, n_p,
                num_t, num_u, num_p,
                n_e1, n_e2, n_e3, 
                t_range, v_range, p_range,
                vbc_point, vbc_point2, vbc2_point, vbc2_point2,
                tq_point, tq_point2, tq_point3,
                l1, l2, n1, n2, n3, m,
                g_vector, g2_vector,
                lut, nuu ):
    fdata = h5py.File( filename, "w" )
    fdata[ "n_f" ] = n_f
    fdata[ "n_t" ] = n_t
    fdata[ "n_u" ] = n_u
    fdata[ "n_p" ] = n_p
    fdata[ "num_t" ] = num_t
    fdata[ "num_u" ] = num_u
    fdata[ "num_p" ] = num_p
    fdata[ "n_e1" ] = n_e1
    fdata[ "n_e2" ] = n_e2
    fdata[ "n_e3" ] = n_e3
    fdata[ "t_range" ] = t_range
    fdata[ "v_range" ] = v_range
    fdata[ "p_range" ] = p_range
    # fdata[ "tbc_control" ] = tbc_control
    # fdata[ "tbc2_control" ] = tbc2_control
    fdata[ "vbc_point" ] = vbc_point
    fdata[ "vbc_point2" ] = vbc_point2
    fdata[ "vbc2_point" ] = vbc2_point
    fdata[ "vbc2_point2" ] = vbc2_point2
    fdata[ "tq_point" ] = tq_point
    fdata[ "tq_point2" ] = tq_point2
    fdata[ "tq_point3" ] = tq_point3
    fdata[ "l1" ] = l1
    fdata[ "l2" ] = l2
    fdata[ "n1" ] = n1
    fdata[ "n2" ] = n2
    fdata[ "n3" ] = n3
    fdata[ "m" ] = m
    fdata[ "g_vector" ] = g_vector
    fdata[ "g2_vector" ] = g2_vector

    with open( filename_lut, "w" ) as lut_output:
        pickle.dump( lut, lut_output )

    with open( filename_nuu, "w" ) as nuu_output:
        pickle.dump( nuu, nuu_output )


import ipdb; ipdb.set_trace()
store_data( "./test_geo/model_new2_lin.data", "./test_geo/model_new2_lut.data",
            "./test_geo/model_new2_nuu.data",
            n_f, n_t, n_u, n_p,
            num_t, num_u, num_p,
            n_e1, n_e2, n_e3,
            t_range, v_range, p_range,
            vbc_point, vbc_point2, vbc2_point, vbc2_point2,
            tq_point, tq_point2, tq_point3,
            l1, l2, n1, n2, n3, m,
            g_vector, g2_vector,
            lut, nuu )

#######
### Check the data stored
#######
'''
def retrieve_data( filename, filename_lut, filename_nuu ):
    fdata = h5py.File( filename, "r" )
    n_f = fdata[ "n_f" ].value
    n_t = fdata[ "n_t" ].value
    n_u = fdata[ "n_u" ].value
    n_p = fdata[ "n_p" ].value
    num_t = fdata[ "num_t" ].value
    num_u = fdata[ "num_u" ].value
    num_p = fdata[ "num_p" ].value
    n_e1 = fdata[ "n_e1" ].value
    n_e2 = fdata[ "n_e2" ].value
    n_e3 = fdata[ "n_e3" ].value
    n_e4 = fdata[ "n_e4" ].value
    n_e5 = fdata[ "n_e5" ].value
    t_range = fdata[ "t_range" ].value
    v_range = fdata[ "v_range" ].value
    p_range = fdata[ "p_range" ].value
    vbc_point = fdata[ "vbc_point" ].value
    vbc_point2 = fdata[ "vbc_point2" ].value
    l1 = fdata[ "l1" ].value
    l2 = fdata[ "l2" ].value
    n1 = fdata[ "n1" ].value
    n2 = fdata[ "n2" ].value
    n3 = fdata[ "n3" ].value
    m = fdata[ "m" ].value
    g_vector = fdata[ "g_vector" ].value

    with open( filename_lut, "rb" ) as data_lut:
        lut = pickle.load( data_lut )
    with open( filename_nuu, "rb" ) as data_nuu:
        nuu = pickle.load( data_nuu )

    return ( n_f, n_t, n_u, n_p,
             num_t, num_u, num_p,
             n_e1, n_e2, n_e3, n_e4, n_e5,
             t_range, v_range, p_range, vbc_point, vbc_point2,
             l1, l2, n1, n2, n3, m, g_vector,
             lut, nuu )


( n_f_r, n_t_r, n_u_r, n_p_r,
  num_t_r, num_u_r, num_p_r,
  n_e1_r, n_e2_r, n_e3_r, n_e4_r, n_e5_r,
  t_range_r, v_range_r, p_range_r, vbc_point_r, vbc_point2_r,
  l1_r, l2_r, n1_r, n2_r, n3_r, m_r, g_vector_r,
  lut_r, nuu_r ) = retrieve_data("model_new_lin.data", "model_new_lut.data",
                                 "model_new_nuu.data")

for i in range(n_u):
    if ( (nuu[i]-nuu_r[i]).min()>0 ) or ( (nuu[i]-nuu_r[i]).max()>0 ) :
        print "nuu tensor has problem."

for i in range(n_t):
    if ( (lut[i]-lut_r[i]).min()>0 ) or ( (lut[i]-lut_r[i]).max() > 0.0 ) :
        print "lut tensor has problem."
'''
