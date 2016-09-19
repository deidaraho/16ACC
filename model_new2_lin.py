import dolfin as dl
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
# from optwrapper import nlp, npsol, snopt
import optwrapper as ow
from datetime import date
import os
import math
import sys
import h5py
import pickle

'''
1. Using the same strategy in model7s.py
2. Using sparse matrix linear algebra in Scipy
3. The new OP geometry is imported from Gmsh
4. The linearized model from Burn's
'''

def retrieve_data( filename ):
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
    t_range = fdata[ "t_range" ].value
    v_range = fdata[ "v_range" ].value
    p_range = fdata[ "p_range" ].value
    vbc_point = fdata[ "vbc_point" ].value
    vbc_point2 = fdata[ "vbc_point2" ].value
    vbc2_point = fdata[ "vbc2_point" ].value
    vbc2_point2 = fdata[ "vbc2_point2" ].value
    # tq_point = fdata[ "tq_point" ].value
    # tq_point2 = fdata[ "tq_point2" ].value
    # tq_point3 = fdata[ "tq_point3" ].value
    l1 = fdata[ "l1" ].value
    l2 = fdata[ "l2" ].value
    n1 = fdata[ "n1" ].value
    n2 = fdata[ "n2" ].value
    n3 = fdata[ "n3" ].value
    m = fdata[ "m" ].value
    g_vector = fdata[ "g_vector" ].value
    g2_vector = fdata[ "g2_vector" ].value

    return ( n_f, n_t, n_u, n_p,
             num_t, num_u, num_p,
             n_e1, n_e2, n_e3,
             t_range, v_range, p_range,
             vbc_point, vbc_point2,
             vbc2_point, vbc2_point2,
             # tq_point, tq_point2, tq_point3,
             l1, l2, n1, n2, n3, m,
             g_vector, g2_vector )


( n_f, n_t, n_u, n_p,
  num_t, num_u, num_p,
  n_e1, n_e2, n_e3,
  t_range, v_range, p_range,
  vbc_point, vbc_point2,
  vbc2_point, vbc2_point2,
  # tq_point, tq_point2, tq_point3,
  l1, l2, n1, n2, n3, m,
  g_vector, g2_vector ) = retrieve_data( "model_new2_lin.data" )

tq1_drt = './target_area/tq_point14.npy'
tq2_drt = './target_area/tq_point17.npy'
tq3_drt = './target_area/tq_point18.npy'

part = '(whole)'
tq_point = np.load( tq1_drt )
tq_point2 = np.load( tq1_drt )
tq_point3 = np.load( tq1_drt )

# ###########
# n_f = n_f/3 # MPC with moving target
# ###########
# import ipdb; ipdb.set_trace()
n_total = n_f*( num_t+1+1 ) + num_u + num_p + ( 1 + 1 )*2 # 2 heaters and fans
n_constraint = n_f*n_e1 + n_e2 + n_e3

##############################
## Optimization Code
##############################
tidx = np.arange( 0, n_f*num_t ).reshape( ( n_f, num_t ) ) # temperature indx
uidx = ( tidx.size +
         np.arange( 0, num_u ) ) # velocity indx
pidx = ( tidx.size + uidx.size +
         np.arange( 0, num_p ) ) # pressure indx
vidx = ( tidx.size + uidx.size + pidx.size +
         np.arange( 0, n_f ) ) # heater control, indx
vuidx = ( tidx.size + uidx.size + pidx.size + vidx.size +
          np.arange( 0, 1 ) )   # velocity control 1 of N1, indx
vu2idx = ( tidx.size + uidx.size + pidx.size + vidx.size + vuidx.size +
           np.arange( 0, 1 ) )  # velocity control 2 of N1, indx
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
    
tqidx = [] # index for target area
for i in tq_point:
    tqidx.append( t_range.tolist().index(i) )
tqidx = np.array( tqidx )
tq2idx = [] # indx for target area 2
for i in tq_point2:
    tq2idx.append( t_range.tolist().index(i) )
tq2idx = np.array( tq2idx )
tq3idx = []    # indx for target area 3
for i in tq_point3:
    tq3idx.append( t_range.tolist().index(i) )
tq3idx = np.array( tq3idx )

e1idx = np.arange( 0, n_f*n_e1 ).reshape( ( n_f, n_e1 ) )
e2idx = ( e1idx.size +
          np.arange( 0, n_e2 ) )
e3idx = ( e1idx.size + e2idx.size +
          np.arange( 0, n_e3 ) )

def objf(out, x):
    out[0] = 0.5/n_f * ( 1000.0 * np.sum( ( x[tidx] - z_d ) * ( x[tidx] - z_d ) )
                         + t_final * ( alpha * x[vidx].dot( x[vidx] ) +
                                       alpha * x[v2idx].dot( x[v2idx] ) +
                                       beta * x[vuidx].dot( x[vuidx] ) +
                                       beta * x[v2uidx].dot( x[v2uidx] ) +
                                       beta2 * x[vu2idx].dot( x[vu2idx] ) +
                                       beta2 * x[v2u2idx].dot( x[v2u2idx] ) ) )

def objg(out, x):
    out[tidx] = 1000.0/(n_f) * ( x[tidx] - z_d )
    out[vidx] = 1.0/n_f * t_final * alpha * x[vidx]
    out[v2idx] = 1.0/n_f * t_final * alpha * x[v2idx]
    out[vuidx] = 1.0/n_f * t_final * beta * x[vuidx]
    out[v2uidx] = 1.0/n_f * t_final * beta * x[v2uidx]
    out[vu2idx] = 1.0/n_f * t_final * beta2 * x[vu2idx]
    out[v2u2idx] = 1.0/n_f * t_final * beta2 * x[v2u2idx]

def objf2(out, x):
    out[0] = 0.5/n_f * ( t_final * ( alpha * x[vidx].dot( x[vidx] ) +
                                     beta * x[vuidx].dot( x[vuidx] ) +
                                     beta2 * x[vu2idx].dot( x[vu2idx] ) ) )
    for i in range(n_f):
        out[0] += 0.5/n_f * ( ( 1.0/num_t*np.sum(x[tidx[i,:]]) - z_d ) *
                              ( 1.0/num_t*np.sum(x[tidx[i,:]]) - z_d ) )

def objg2(out, x):
    out[vidx] = 1.0/n_f * t_final * alpha * x[vidx]
    out[vuidx] = 1.0/n_f * t_final * beta * x[vuidx]
    out[vu2idx] = 1.0/n_f * t_final * beta2 * x[vu2idx]
    for i in range(n_f):
        out[tidx[i,:]] = 1.0/n_f * 1.0/num_t * ( np.sum(x[tidx[i,:]])/num_t -
                                                 z_d) * np.ones( (num_t,) )
        
def objgpattern():
    out = np.zeros( ( n_total, ), dtype=np.int )
    out[tidx] = 1
    out[vidx] = 1
    out[v2idx] = 1
    out[vuidx] = 1
    out[v2uidx] = 1
    out[vu2idx] = 1
    out[v2u2idx] = 1
    return out

##########
## cost function with local target
##########
def objf_local(out, x):
    out[0] = 0.5/n_f * ( 1000.0/num_t * np.sum( ( x[tidx[:,tqidx]] - z_d ) * ( x[tidx[:,tqidx]] - z_d ) )
                         + t_final * ( alpha * x[vidx].dot( x[vidx] ) +
                                       alpha * x[v2idx].dot( x[v2idx] ) +
                                       beta * x[vuidx].dot( x[vuidx] ) +
                                       beta * x[v2uidx].dot( x[v2uidx] ) +
                                       beta2 * x[vu2idx].dot( x[vu2idx] ) +
                                       beta2 * x[v2u2idx].dot( x[v2u2idx] ) ) )
    
def objg_local(out, x):
    out[tidx[:,tqidx]] = 1000.0/(n_f*num_t) * ( x[tidx[:,tqidx]] - z_d )
    out[vidx] = 1.0/n_f * t_final * alpha * x[vidx]
    out[v2idx] = 1.0/n_f * t_final * alpha * x[v2idx]
    out[vuidx] = 1.0/n_f * t_final * beta * x[vuidx]
    out[v2uidx] = 1.0/n_f * t_final * beta * x[v2uidx]
    out[vu2idx] = 1.0/n_f * t_final * beta2 * x[vu2idx]
    out[v2u2idx] = 1.0/n_f * t_final * beta2 * x[v2u2idx]
    
def objgpattern_local():
    out = np.zeros( ( n_total, ), dtype=np.int )
    out[tidx[:,tqidx]] = 1
    out[vidx] = 1
    out[v2idx] = 1
    out[vuidx] = 1
    out[vu2idx] = 1
    out[v2uidx] = 1
    out[v2u2idx] = 1
    return out
# local cost for the second target
def objf_local2(out, x):
    out[0] = 0.5/n_f * ( 10.0/num_t * np.sum( ( x[tidx[:,tq2idx]] - z_d ) * ( x[tidx[:,tq2idx]] - z_d ) )
                         + t_final * ( alpha * x[vidx].dot( x[vidx] ) +
                                       alpha * x[v2idx].dot( x[v2idx] ) +
                                       beta * x[vuidx].dot( x[vuidx] ) +
                                       beta * x[v2uidx].dot( x[v2uidx] ) +
                                       beta2 * x[vu2idx].dot( x[vu2idx] ) +
                                       beta2 * x[v2u2idx].dot( x[v2u2idx] ) ) )
    
def objg_local2(out, x):
    out[tidx[:,tq2idx]] = 10.0/(n_f*num_t) * ( x[tidx[:,tq2idx]] - z_d )
    out[vidx] = 1.0/n_f * t_final * alpha * x[vidx]
    out[v2idx] = 1.0/n_f * t_final * alpha * x[v2idx]
    out[vuidx] = 1.0/n_f * t_final * beta * x[vuidx]
    out[v2uidx] = 1.0/n_f * t_final * beta * x[v2uidx]
    out[vu2idx] = 1.0/n_f * t_final * beta2 * x[vu2idx]
    out[v2u2idx] = 1.0/n_f * t_final * beta2 * x[v2u2idx]
    
def objgpattern_local2():
    out = np.zeros( ( n_total, ), dtype=np.int )
    out[tidx[:,tq2idx]] = 1
    out[vidx] = 1
    out[v2idx] = 1
    out[vuidx] = 1
    out[vu2idx] = 1
    out[v2uidx] = 1
    out[v2u2idx] = 1
    return out

# local cost for the third target
def objf_local3(out, x):
    out[0] = 0.5/n_f * ( 10.0/num_t * np.sum( ( x[tidx[:,tq3idx]] - z_d ) * ( x[tidx[:,tq3idx]] - z_d ) )
                         + t_final * ( alpha * x[vidx].dot( x[vidx] ) +
                                       alpha * x[v2idx].dot( x[v2idx] ) +
                                       beta * x[vuidx].dot( x[vuidx] ) +
                                       beta * x[v2uidx].dot( x[v2uidx] ) +
                                       beta2 * x[vu2idx].dot( x[vu2idx] ) +
                                       beta2 * x[v2u2idx].dot( x[v2u2idx] ) ) )
    
def objg_local3(out, x):
    out[tidx[:,tq3idx]] = 10.0/(n_f*num_t) * ( x[tidx[:,tq3idx]] - z_d )
    out[vidx] = 1.0/n_f * t_final * alpha * x[vidx]
    out[v2idx] = 1.0/n_f * t_final * alpha * x[v2idx]
    out[vuidx] = 1.0/n_f * t_final * beta * x[vuidx]
    out[v2uidx] = 1.0/n_f * t_final * beta * x[v2uidx]
    out[vu2idx] = 1.0/n_f * t_final * beta2 * x[vu2idx]
    out[v2u2idx] = 1.0/n_f * t_final * beta2 * x[v2u2idx]
    
def objgpattern_local3():
    out = np.zeros( ( n_total, ), dtype=np.int )
    out[tidx[:,tq3idx]] = 1
    out[vidx] = 1
    out[v2idx] = 1
    out[vuidx] = 1
    out[vu2idx] = 1
    out[v2uidx] = 1
    out[v2u2idx] = 1
    return out

# moving cost function
def objf2_local(out, x):
    out[0] = 0.5/n_f * ( 100.0/num_t * np.sum( ( x[tidx[0:n_f/3,tqidx]] - z_d ) * ( x[tidx[0:n_f/3,tqidx]] - z_d ) ) +
                         100.0/num_t * np.sum( ( x[tidx[n_f/3:2*n_f/3,tq2idx]] - z_d ) * ( x[tidx[n_f/3:2*n_f/3,tq2idx]] - z_d ) ) +
                         100.0/num_t * np.sum( ( x[tidx[2*n_f/3:,tq3idx]] - z_d ) * ( x[tidx[2*n_f/3:,tq3idx]] - z_d ) ) +
                         + t_final * ( alpha * x[vidx].dot( x[vidx] ) +
                                       alpha * x[v2idx].dot( x[v2idx] ) +
                                       beta * x[vuidx].dot( x[vuidx] ) +
                                       beta * x[v2uidx].dot( x[v2uidx] ) +
                                       beta2 * x[vu2idx].dot( x[vu2idx] ) +
                                       beta2 * x[v2u2idx].dot( x[v2u2idx] ) ) )

def objg2_local(out, x):
    out[tidx[0:n_f/3,tqidx]] = 100.0/(n_f*num_t) * ( x[tidx[0:n_f/3,tqidx]] - z_d )
    out[tidx[n_f/3:2*n_f/3,tq2idx]] = 100.0/(n_f*num_t) * ( x[tidx[n_f/3:2*n_f/3,tq2idx]] - z_d )
    out[tidx[2*n_f/3:,tq3idx]] = 100.0/(n_f*num_t) * ( x[tidx[2*n_f/3:,tq3idx]] - z_d )
    out[vidx] = 1.0/n_f * t_final * alpha * x[vidx]
    out[v2idx] = 1.0/n_f * t_final * alpha * x[v2idx]
    out[vuidx] = 1.0/n_f * t_final * beta * x[vuidx]
    out[v2uidx] = 1.0/n_f * t_final * beta * x[v2uidx]
    out[vu2idx] = 1.0/n_f * t_final * beta2 * x[vu2idx]
    out[v2u2idx] = 1.0/n_f * t_final * beta2 * x[v2u2idx]
    
def objgpattern2_local ():
    out = np.zeros( ( n_total, ), dtype=np.int )
    out[tidx[0:n_f/3,tqidx]] = 1
    out[tidx[n_f/3:2*n_f/3,tq2idx]] = 1
    out[tidx[2*n_f/3:,tq3idx]] = 1
    out[vidx] = 1
    out[v2idx] = 1
    out[vuidx] = 1
    out[v2uidx] = 1
    out[vu2idx] = 1
    out[v2u2idx] = 1
    return out    

def consf(out, x):
    x_u = np.zeros(n_u)
    x_p = np.zeros(n_p)
    x_u[v_range] = x[uidx]
    x_u[vbc_point] = x[vuidx]
    x_u[vbc_point2] = x[vu2idx]
    x_u[vbc2_point] = x[v2uidx]
    x_u[vbc2_point2] = x[v2u2idx]
    x_p[p_range] = x[pidx]

    # Equation 1
    for i in range(n_f):
        x_t = np.zeros(n_t)
        x_t_prev = np.zeros(n_t)
        x_t[t_range] = x[tidx[i,:]]
        if i > 0:
            x_t_prev[t_range] = x[tidx[i-1,:]]
        out[e1idx[i,:]] = ( l1[t_range,:].dot( x_t-x_t_prev ) +
                            l2[t_range,:].dot( x_t ) -
                            x[vidx[i]]*g_vector[t_range] -
                            x[v2idx[i]]*g2_vector[t_range] )
    # Equation 2, velocity stationary
    out[e2idx] = ( n1[v_range,:].dot( x_u ) +
                   n2[v_range,:].dot( x_u ) +
                   n3[v_range,:].dot( x_p ) )
    # Equation 3, pressure stationary
    out[e3idx] = np.dot( m, x_u )[p_range]

    
def consg(out, x):
    x_u = np.zeros(n_u)
    x_p = np.zeros(n_p)
    x_u[v_range] = x[uidx]
    x_u[vbc_point] = x[vuidx]
    x_u[vbc_point2] = x[vu2idx]
    x_u[vbc2_point] = x[v2uidx]
    x_u[vbc2_point2] = x[v2u2idx]
    x_p[p_range] = x[pidx]

    # Equation 1
    for k in range(n_f): # time loop
        x_t = np.zeros(n_t)
        x_t[t_range] = x[tidx[k,:]]
        # temperature
        out[np.ix_( e1idx[k,:],tidx[k,:] )] = ( l1[np.ix_( t_range,t_range )] +
                                                l2[np.ix_( t_range,t_range )] )
        if k>0:
            out[np.ix_( e1idx[k,:],tidx[k-1,:] )] = -l1[np.ix_( t_range,t_range )]
    # heat controls
        out[e1idx[k,:], vidx[k]] = -1.0 * g_vector[t_range]
        out[e1idx[k,:], v2idx[k]] = -1.0 * g2_vector[t_range]

    # Equation 2
    # pressure
    out[np.ix_( e2idx,pidx )] = n3[np.ix_( v_range,p_range )]
    # velocity
    out[np.ix_( e2idx,uidx )] = n1[np.ix_( v_range,v_range )] + n2[np.ix_( v_range,v_range )]
    out[e2idx, vuidx[0]] = np.dot( ( n1[np.ix_( v_range,vbc_point )] +
                                     n2[np.ix_( v_range,vbc_point )]),
                                     np.ones( len(vbc_point) ) )
    out[e2idx, vu2idx[0]] = np.dot( (n1[np.ix_( v_range,vbc_point2 )] +
                                     n2[np.ix_( v_range,vbc_point2 )]),
                                     np.ones( len(vbc_point2) ))
    out[e2idx, v2uidx[0]] = np.dot( ( n1[np.ix_( v_range,vbc2_point )] +
                                     n2[np.ix_( v_range,vbc2_point )]),
                                     np.ones( len(vbc2_point) ) )
    out[e2idx, v2u2idx[0]] = np.dot( (n1[np.ix_( v_range,vbc2_point2 )] +
                                     n2[np.ix_( v_range,vbc2_point2 )]),
                                     np.ones( len(vbc2_point2) ) )

    # Equation 3
    # velocity
    out[np.ix_( e3idx, uidx )] = m[np.ix_( p_range,v_range )]
    for i in range(n_e3):
        # velocity control vu
        out[e3idx[i], vuidx[0]] = np.dot( m[p_range[i],vbc_point],
                                          np.ones(len(vbc_point)) )
        # velocity control vu2
        out[e3idx[i], vu2idx[0]] = np.dot( m[p_range[i],vbc_point2],
                                           np.ones(len(vbc_point2)) )
        # velocity control v2u
        out[e3idx[i], v2uidx[0]] = np.dot( m[p_range[i],vbc2_point],
                                          np.ones(len(vbc2_point)) )
        # velocity control v2u2
        out[e3idx[i], v2u2idx[0]] = np.dot( m[p_range[i],vbc2_point2],
                                           np.ones(len(vbc2_point2)) )

def consgpattern():
    out = np.zeros(( n_constraint, n_total ), dtype=np.int)
    # Equation 1
    for i in range(n_f):
        out[np.ix_( e1idx[i,:], tidx[i,:] )] = 1
        if i>0:
            out[np.ix_( e1idx[i,:], tidx[i-1,:] )] = 1
        out[ e1idx[i,:], vidx[i] ] = 1
        out[ e1idx[i,:], v2idx[i]] = 1
        
    # Equation 2
    out[ np.ix_( e2idx, uidx ) ] = 1
    out[ np.ix_( e2idx, pidx ) ] = 1
    out[ e2idx, vuidx[0] ] = 1
    out[ e2idx, vu2idx[0] ] = 1
    out[ e2idx, v2uidx[0] ] = 1
    out[ e2idx, v2u2idx[0] ] = 1
    # Equation 3
    out[np.ix_( e3idx, uidx )] = 1
    out[e3idx, vuidx[0]] = 1
    out[e3idx, vu2idx[0]] = 1
    out[e3idx, v2uidx[0]] = 1
    out[e3idx, v2u2idx[0]] = 1
    return out


########
# assign the problem
########
prob = ow.nlp.SparseProblem(N=n_total, Ncons=n_constraint)
# Added box constraints
x_bnd = np.ones(n_total)
x_bnd_l = -np.inf * x_bnd
x_bnd_h = np.inf * x_bnd
# set heater control range
x_bnd_l[vidx] = 0.0
x_bnd_h[vidx] = 3.0
# set fan's control range in two directions
x_bnd_l[vuidx[0]] = 0.1
x_bnd_h[vuidx[0]] = 2.0
x_bnd_l[vu2idx[0]] = -2.0
x_bnd_h[vu2idx[0]] = 0.0
x_bnd_l[v2uidx[0]] = -2.0
x_bnd_h[v2uidx[0]] = -0.1
x_bnd_l[v2u2idx[0]] = -2.0
x_bnd_h[v2u2idx[0]] = 0.0
prob.consBox( x_bnd_l, x_bnd_h ) # set box constraint
# constraints' range
consf_bounds_low = np.zeros( (n_constraint,) )
consf_bounds_upper = np.zeros( (n_constraint,) )
prob.initPoint( np.ones(n_total) )
# cost function parameters
z_d = 1.0 # target temperature
alpha = 3E-2 # coefficients for running cost
beta = 1E-2
beta2 = 1E-2
# w_alpha = np.array( range(n_f), dtype='float' )/( n_f-1 )
w_alpha = np.ones( (n_f,) )
t_start = np.zeros( (num_t,) )
t_final = 300
# cost function
prob.objFctn( objf )
prob.objGrad( objg, pattern=objgpattern() )
prob.consFctn( consf, lb=consf_bounds_low, ub=consf_bounds_upper )
prob.consGrad( consg, pattern=consgpattern() )

solver = ow.ipopt.Solver( prob ) ## change this line to use another solver
solver.debug = True
tdy = date.today()
# part = '(local3)'
drt = './results/modelNew2_lin/'+str(tdy) + part
if not os.path.exists(drt):
    os.makedirs(drt)

# solver.options[ "printFile" ] = (drt + "/optwrp1.txt")
# solver.options[ "printLevel" ] = 10
# solver.options[ "minorPrintLevel" ] = 10
# solver.options["qpSolver"] = "cg"
# solver.options[ "summaryFile" ] = "debugs.txt"

import ipdb; ipdb.set_trace()

# if( not prob.checkGrad( debug=True ) ):
#     sys.exit( "Gradient check failed." )

# if( not prob.checkPattern( debug=True ) ):
#     sys.exit( "Pattern check failed." )

solver.solve()

print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Retval: " + str( prob.soln.retval ) )

def store_result( filename_fin, filename_txt,
                  final_array,
                  alpha, beta, beta2,
                  value, retval):
    np.save( filename_fin, final_array )
    expfile = open( filename_txt, "w" )
    expfile.write( str(tdy) + '\n' )
    expfile.write( ( "The alpha is: "+ str(alpha) + ", and the beta is: " + str(beta) + ", and the beta2 is: " + str(beta2)  + '\n' ) )
    expfile.write( "Value: " + str( value ) + '\n' )
    expfile.write( "Retval: " + str( retval ) + '\n' )
    expfile.write( prob.soln.getStatus() + '\n' )

store_result( (drt+'/results1'), (drt+'/exp1.txt'),
              prob.soln.final,
              alpha, beta, beta2,
              prob.soln.value, prob.soln.retval)

import ipdb; ipdb.set_trace()
########
# assign the problem for the second period
########
prob2 = nlp.SparseProblem(N=n_total, Ncons=n_constraint)
prob2.consBox( x_bnd_l, x_bnd_h ) # set box constraint
t_start = prob.soln.final[tidx[-1,:]]

prob2.objFctn( objf )
prob2.objGrad( objg, pattern=objgpattern() )
prob2.consFctn( consf, lb=consf_bounds_low, ub=consf_bounds_upper )
prob2.consGrad( consg, pattern=consgpattern() )

solver2 = snopt.Solver( prob2 ) ## change this line to use another solver
solver2.debug = True
solver2.printOpts[ "printLevel" ] = 10
solver2.printOpts["printFile"] = (drt + "/optwrp2.txt")
solver2.printOpts["minorPrintLevel"] = 10
solver2.printOpts["printLevel"] = 10
solver2.solveOpts["qpSolver"] = "cg"
solver2.solve()
print( prob2.soln.getStatus() )
print( "Value: " + str( prob2.soln.value ) )
print( "Retval: " + str( prob2.soln.retval ) )

store_result( (drt+'/results2'), (drt+'/exp2.txt'),
              prob2.soln.final,
              alpha, beta, beta2,
              prob2.soln.value, prob2.soln.retval)

########
# assign the problem for the third period
########
prob3 = nlp.SparseProblem(N=n_total, Ncons=n_constraint)
prob3.consBox( x_bnd_l, x_bnd_h ) # set box constraint
t_start = prob2.soln.final[tidx[-1,:]]
# cost function
prob3.objFctn( objf )
prob3.objGrad( objg, pattern=objgpattern() )
prob3.consFctn( consf, lb=consf_bounds_low, ub=consf_bounds_upper )
prob3.consGrad( consg, pattern=consgpattern() )

solver3 = snopt.Solver( prob3 ) ## change this line to use another solver
solver2.debug = True
solver3.printOpts[ "printLevel" ] = 10
solver3.printOpts["printFile"] = (drt + "/optwrp3.txt")
solver3.printOpts["minorPrintLevel"] = 10
solver3.printOpts["printLevel"] = 10
solver3.solveOpts["qpSolver"] = "cg"
solver3.solve()
print( prob3.soln.getStatus() )
print( "Value: " + str( prob3.soln.value ) )
print( "Retval: " + str( prob3.soln.retval ) )

store_result( (drt+'/results3'), (drt+'/exp3.txt'),
              prob3.soln.final,
              alpha, beta, beta2,
              prob3.soln.value, prob3.soln.retval)
