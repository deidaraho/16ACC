from dolfin import *
from dolfin import *
import numpy as np
import scipy.sparse as sp
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from datetime import date
import os
import math
import sys
import h5py
import pickle
import ipdb

# set_log_level(1)
# drt = "./results/modelNew2_lin/2015-09-29(moving)"

# drt = "./results/modelNew2/2016-03-02(local09)"

drt = "./results/modelNew2_lin/2016-03-01(local15)"

mesh = Mesh("./test_geo/test2.xml")
subdomains = MeshFunction("size_t", mesh, "./test_geo/test2_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "./test_geo/test2_facet_region.xml")

V0 = FunctionSpace(mesh,"DG",0)
V = VectorFunctionSpace(mesh,"CG",2)
P = FunctionSpace(mesh, "CG", 1)
W = V * P

##############
## Got the data from the Optimization
##############
t_final = 300
dt = 10
time_axis = range(0,t_final+dt,dt)
time_axis = np.array(time_axis)
def retrieve_result( filename_lin,
                     filename_lut,
                     filename_final ):
    fdata = h5py.File( filename_lin, "r" )
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
    t_range = fdata[ "t_range" ].value
    v_range = fdata[ "v_range" ].value
    p_range = fdata[ "p_range" ].value
    vbc_point = fdata[ "vbc_point" ].value
    vbc_point2 = fdata[ "vbc_point2" ].value
    vbc2_point = fdata[ "vbc2_point" ].value
    vbc2_point2 = fdata[ "vbc2_point2" ].value

    l1 = fdata[ "l1" ].value
    l2 = fdata[ "l2" ].value
    g_vector = fdata[ "g_vector" ].value
    g2_vector = fdata[ "g2_vector" ].value
    # import ipdb; ipdb.set_trace()
    with open( filename_lut, "rb" ) as data_lut:
        lut = pickle.load( data_lut )

    final_array = np.load( filename_final )

    return ( n_f, n_t, n_u, n_p,
             num_t, num_u, num_p,
             n_e1, n_e2, n_e3,
             t_range, v_range, p_range,
             vbc_point, vbc_point2,
             vbc2_point, vbc2_point2,
             # tq_point, tq_point2, tq_point3,
             l1, l2,
             g_vector, g2_vector,
             lut, final_array )

( n_f, n_t, n_u, n_p,
  num_t, num_u, num_p,
  n_e1, n_e2, n_e3,
  t_range, v_range, p_range,
  vbc_point, vbc_point2,
  vbc2_point, vbc2_point2,
  # tq_point, tq_point2, tq_point3,
  lt1, lt2,
  g_vector, g2_vector,
  lut, final_array ) = retrieve_result( "model_new2_lin.data",
                                        "model_new2_lut.data",
                                        (drt + "/results1.npy") )

n_total = n_f*( num_t+1+1 ) + num_u + num_p + ( 1 + 1 )*2
n_constraint = n_f*n_e1 + n_e2 + n_e3

tidx = np.arange( 0, n_f*num_t ).reshape( ( n_f, num_t ) ) # temperature indx
uidx = ( tidx.size +
         np.arange( 0, num_u ) ) # velocity indx
pidx = ( tidx.size + uidx.size +
         np.arange( 0, num_p ) ) # pressure indx
vidx = ( tidx.size + uidx.size + pidx.size +
         np.arange( 0, n_f ) ) # heater control, indx
vuidx = ( tidx.size + uidx.size + pidx.size + vidx.size +
          np.arange( 0, 1 ) )   # velocity control 1, indx
vu2idx = ( tidx.size + uidx.size + pidx.size + vidx.size + vuidx.size +
           np.arange( 0, 1 ) )  # velocity control 2, indx
v2idx = ( tidx.size + uidx.size + pidx.size +
          vidx.size + vuidx.size + vu2idx.size +
          np.arange( 0, n_f ) )  # heater control, indx
v2uidx = ( tidx.size + uidx.size + pidx.size +
           vidx.size + vuidx.size + vu2idx.size + v2idx.size +
           np.arange(0,1) )      # velocity control 1 of N2, indx
v2u2idx = ( tidx.size + uidx.size + pidx.size +
           vidx.size + vuidx.size + vu2idx.size +
           v2idx.size + v2uidx.size +
           np.arange(0,1) )      # velocity control 2 of N2, indx

e1idx = np.arange( 0, n_f*n_e1 ).reshape( ( n_f, n_e1 ) )
e2idx = ( e1idx.size +
          np.arange( 0, n_e2 ) )
e3idx = ( e1idx.size + e2idx.size +
          np.arange( 0, n_e3 ) )

finalT = np.zeros( (n_f+1,n_t) )
for i in range(1,n_f+1):
    finalT[ i,t_range ] = final_array[tidx[i-1,:]]

finalU = np.zeros( (n_u,) )
finalU[v_range] = final_array[uidx]
finalU[vbc_point] = final_array[vuidx]
finalU[vbc_point2] = final_array[vu2idx]
finalU[vbc2_point] = final_array[v2uidx]
finalU[vbc2_point2] = final_array[v2u2idx]

finalP = np.zeros( (n_p,) )
finalP[p_range] = final_array[pidx]

finalV = np.zeros( (n_f+1,) )
finalV = final_array[vidx]
finalV2 = final_array[v2idx]

finalVU = final_array[vuidx]
finalVU2 = final_array[vu2idx]
finalV2U = final_array[v2uidx]
finalV2U2 = final_array[v2u2idx]

##############
##### FEM simulation
##############
# ke, velocity distinguishing coefficient 
ke = Function(V0)
ke_values = [0.5, 100]  # values of re in the two subdomains
for cell_no in range(len(subdomains.array())):
    if subdomains.array()[cell_no] == 34: # wall region
        ke.vector()[cell_no] = ke_values[1]
    else:
        ke.vector()[cell_no] = ke_values[0]

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
        
# import ipdb;ipdb.set_trace()
####
#### Stokes/ Navier-Stokes
####
re = Constant(0.1)

# Define test and trail functions
(v,q) = TestFunctions(W)
w = Function(W)
(u,p) = split(w)
F = ( ke*inner(u,v) + re*inner(grad(u), grad(v)) + inner(grad(u)*u, v) - div(v)*p + q*div(u) )*dx
F_asm = assemble(F)
F_mat = F_asm.array()
f = Constant((0,0))
L = inner(f,v)*dx
L_asm = assemble(L)

# Boundary conditions
v1u1 = finalVU
v1u2 = finalVU2
v2u1 = finalV2U
v2u2 = finalV2U2
# import ipdb; ipdb.set_trace()
bc1 = DirichletBC( W.sub(0), Constant( ( v1u1, 0.0 ) ), boundaries, 36 )
bc2 = DirichletBC( W.sub(0), Constant( ( 0.0,0.0 ) ), boundaries, 39 )
bc3 = DirichletBC( W.sub(0), Constant( ( v2u1, 0.0 ) ), boundaries, 37 )
bc4 = DirichletBC( W.sub(1), Constant(0), boundaries, 38 )
bcs = [ bc1,bc2,bc3,bc4 ]
solve(F==0,w,bcs)
# visualize
# u,p = w.split()
u = Function(V)
p = Function(P)
assign(u, w.sub(0))
assign(p,w.sub(1))

dolU = u.vector().array()
dolP = p.vector().array()

abs_errorU = abs( dolU - finalU ) / np.linalg.norm( dolU )
abs_errorP = abs( dolP - finalP ) / np.linalg.norm( dolP )
# average velocity error
errU_avg = np.linalg.norm( dolU - finalU ) / np.linalg.norm( dolU )
errU_std = np.std( dolU - finalU )/np.linalg.norm( dolU )

# average pressure error
errP_avg = np.linalg.norm( dolP - finalP )/np.linalg.norm( dolP )
errP_std = np.std( dolP - finalP )/ np.linalg.norm( dolP )

print "error U: " + str( errU_avg )
print "std P: " + str( errU_std )
print "error P: " + str( errP_avg )
print "std P: " + str( errP_std )
# ipdb.set_trace()
##### temperature
T = FunctionSpace(mesh, "CG", 1)
te = TrialFunction(T)
de = TestFunction(T)
te0 = Function(T)
te1 = Function(T)

# Time-stepping
t = 10
dt = 10

t = 0.1
dt = 0.1

k_dt = Constant(dt)
Tf = 300

g = Expression('x[0] >= 0.5 & x[0] <= 2.0 & x[1] >= 3.0 & x[1] <= 4.5 ? 1.0 : 0') # heater position
g2 = Expression('x[0] >= 8.0 & x[0] <= 9.5 & x[1] >= 3.0 & x[1] <= 4.5 ? 1.0 : 0') # heater position 2
# import ipdb;ipdb.set_trace()
# Create files for storing solution
a1 = (1/k_dt)*inner(te,de)*dx + ke2*inner(grad(te),grad(de))*dx + inner(grad(te),u)*de*dx
low_g = Constant( finalV[0] )
low_g2 = Constant( finalV2[0] )
l1 = (1/k_dt)*inner(te0,de)*dx + low_g*g*de*dx + low_g2*g2*de*dx
A1 = assemble(a1)

# solve temperature
tbc1 = DirichletBC(T, Constant(0.0), boundaries, 36)
tbc2 = DirichletBC(T, Constant(0.0), boundaries, 37)
tbc3 = DirichletBC(T, Constant(0.0), boundaries, 38)
tbc4 = DirichletBC(T, Constant(0.0), boundaries, 39)
tbcs = [tbc1,tbc2,tbc3,tbc4]

# dolT = np.zeros( (n_f+1,n_t) )
i = 0
#######
## fine the time step
#######
finalV1_x = np.zeros( (Tf/dt,) )
finalV2_x = np.zeros( (Tf/dt,) )
for i in range( 100 ):
    finalV1_x[i::100] = finalV
    finalV2_x[i::100] = finalV2

i = 0
while t <= Tf:
    # Compute tentative velocity step
    low_g = Constant( finalV1_x[i] )
    low_g2 = Constant( finalV2_x[i] )
    b1 = assemble(l1)
    for bco in tbcs:
        bco.apply(A1,b1)
    solve(A1, te1.vector(), b1)
    end()
    # viz_t = plot(te1, title="Temperature", rescale=True, interactive=True)
    # dolT[i+1,:] = te1.vector().array()
    te0.assign(te1)
    t += dt
    i += 1
    # if i > 298:
    #     ipdb.set_trace()

dolT = te1.vector().array()

###############
#####
l3 = np.zeros( (len( t_range ), n_e1 ) )
for i in range( n_e1 ):
    l3[i,:] = lut[t_range[i]].dot( dolU )[t_range]
lt_inv = np.linalg.inv( lt1[ np.ix_( t_range, t_range ) ]
                        + lt2[ np.ix_( t_range, t_range ) ]
                        + l3 )
x_t_pre = np.zeros( ( len(t_range), ) )

t = 10
dt = 10
i = 0
while t <= Tf:
    x_t = lt_inv.dot( lt1[ np.ix_( t_range, t_range ) ].dot( x_t_pre )
                      + finalV[i] * g_vector[t_range]
                      + finalV2[i] * g2_vector[t_range] )
    i = i+1
    t = t+dt
    x_t_pre = x_t

dolT = np.zeros( (n_t,) )
dolT[t_range] = x_t
# need a way mapping T2 to T1
# ipdb.set_trace()
abs_errorT = abs( finalT[-1,:] - dolT )
print "error T: " + str( abs_errorT.mean() )
print "std T: " + str( abs_errorT.std() )

np.save( ( drt+'/tem_dol' ), dolT )
ipdb.set_trace()
###############
# saving
###############

np.save( ( drt+'/tem_error_new' ), abs_errorT )
###############
# plotting
###############
# plot final-time temperature error
nx = 100
ny = 100
X = T.dofmap().tabulate_all_coordinates(mesh)
X.resize( (T.dim(),2) )
x_cor = X[:,0]
y_cor = X[:,1]
xi, yi = np.linspace(x_cor.min(), x_cor.max(), nx+1), np.linspace(y_cor.min(), y_cor.max(), ny+1)
xi, yi = np.meshgrid(xi, yi)

levels = MaxNLocator(nbins=15).tick_values(abs_errorT.min(), abs_errorT.max())
cmap = plt.get_cmap('Reds')

fig = plt.figure()
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
ax = fig.add_subplot(111, aspect="equal")

temp_T = abs_errorT
rbf = scipy.interpolate.Rbf(x_cor, y_cor, temp_T, function='linear')
temp_zi = rbf(xi, yi)
CS = ax.contourf(xi, yi, temp_zi, levels=levels, cmap=cmap)
CS2 = ax.contour(CS, levels=CS.levels, colors = 'r', hold='on')
cbar = fig.colorbar(CS)
cbar.add_lines(CS2)
CS.ax.axes.get_xaxis().set_visible(False)
CS.ax.axes.get_yaxis().set_visible(False)
CS2.ax.axes.get_xaxis().set_visible(False)
CS2.ax.axes.get_yaxis().set_visible(False)
fig.savefig((drt + '/temperature_error_new.pdf'), dpi=1000, format='pdf')
plt.close()

import ipdb; ipdb.set_trace()
# plot velocity
plt.figure()
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
XU = V.dofmap().tabulate_all_coordinates(mesh)
v_dim = V.dim()
XU.resize((V.dim(),2))
xu_cor = XU[::2,0]
yu_cor = XU[::2,1]
dx = 0.1
dy = 0.1
( xm, ym ) = np.meshgrid( np.arange( xu_cor.min(), xu_cor.max(), dx ),
                          np.arange( yu_cor.min(), yu_cor.max(), dy ) )
# linear interplation
u_x = finalU[::2] - dolU[::2]
u_y = finalU[1::2] - dolU[1::2]
Ux = scipy.interpolate.Rbf(xu_cor, yu_cor, u_x, function='linear')
Uy = scipy.interpolate.Rbf(xu_cor, yu_cor, u_y, function='linear')
u_xi = Ux(xm, ym)
u_yi = Uy(xm, ym)
q_plot = plt.quiver( xm, ym, u_xi, u_yi, pivot = 'mid', color = 'b' )
q_plot.ax.axes.get_xaxis().set_visible(False)
q_plot.ax.axes.get_yaxis().set_visible(False)
qk = plt.quiverkey(q_plot, 0.1, 0.03, 1, r'$1 \frac{m}{s}$', fontproperties={'weight': 'bold', 'size':20})
plt.savefig((drt + '/velocity_error.pdf'), dpi=1000, format='pdf')
plt.close()

# plot pressure error
plt.figure()
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
XQ = P.dofmap().tabulate_all_coordinates(mesh)
XQ.resize((T.dim(),2))
xq_cor = XQ[:,0]
yq_cor = XQ[:,1]
temp_P = abs( finalP - dolP )
rbf_p = scipy.interpolate.Rbf(xq_cor, yq_cor, temp_P, function='linear')
temp_zi = rbf_p(xi, yi)
cmap = plt.get_cmap('Blues')
levels = MaxNLocator(nbins=15).tick_values(finalP.min(), finalP.max())
CS = plt.contourf(xi, yi, temp_zi, levels=levels, cmap=cmap)
plt.colorbar()
CS.ax.axes.get_xaxis().set_visible(False)
CS.ax.axes.get_yaxis().set_visible(False)
plt.savefig( ( drt + '/pressure_error.pdf' ), dpi=1000, format='pdf' )
plt.close()
