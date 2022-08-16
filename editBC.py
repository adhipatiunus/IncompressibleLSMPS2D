#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 21:57:28 2022

@author: adhipatiunus
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import threading
from joblib import Parallel, delayed

from generate_particle import generate_particles, generate_particles_singleres
from neighbor_search import neighbor_search_cell_list
from neighbor_search_verlet import multiple_verlet
from visualize import visualize

n_thread = threading.active_count()
#%%
def calculate_weight(r_ij, R_e):
    if r_ij < R_e:
        w_ij = (1 - r_ij / R_e)**2
    else:
        w_ij = 0
    return w_ij

def LSMPSb(node_x, node_y, index, Rmax, R, r_e, R_s, neighbor, n_neighbor):
    N = len(node_x)
    n = len(index)
    EtaDx   = sparse.lil_matrix((n,N), dtype=np.float64)
    EtaDy   = sparse.lil_matrix((n,N), dtype=np.float64)
    EtaDxx  = sparse.lil_matrix((n,N), dtype=np.float64)
    EtaDxy  = sparse.lil_matrix((n,N), dtype=np.float64)
    EtaDyy  = sparse.lil_matrix((n,N), dtype=np.float64)
    
    k = 0
    for i in index:
        H_rs = np.zeros((6,6), dtype=np.float64)
        M = np.zeros((6,6), dtype=np.float64)
        P = np.zeros((6,1), dtype=np.float64)
        b_temp = np.zeros((n_neighbor[i], 6, 1), dtype=np.float64)
        
       # print('Calculating derivative for particle ' + str(i) + '/' + str(N))
        
        neighbor_idx = neighbor[i]
        
        #R_max = np.max(R[neighbor_idx])
        
        idx_i = i
        x_i = node_x[idx_i]
        y_i = node_y[idx_i]
        R_e = r_e * Rmax[idx_i]
        R_i = R[idx_i]
                
        H_rs[0, 0] = 1
        H_rs[1, 1] = R_s[i]**-1
        H_rs[2, 2] = R_s[i]**-1
        H_rs[3, 3] = 2 * R_s[i]**-2
        H_rs[4, 4] = R_s[i]**-2
        H_rs[5, 5] = 2 * R_s[i]**-2
                
        for j in range(n_neighbor[i]):
            idx_j = neighbor_idx[j]
            x_j = node_x[idx_j]
            y_j = node_y[idx_j]
            R_j = R[idx_j]
            
            x_ij = x_j - x_i
            y_ij = y_j - y_i
            r_ij = np.sqrt((x_ij)**2 + (y_ij)**2)
             
            p_x = x_ij / R_s[i]
            p_y = y_ij / R_s[i]
            
            P[0, 0] = 1.0
            P[1, 0] = p_x
            P[2, 0] = p_y
            P[3, 0] = p_x**2
            P[4, 0] = p_x * p_y
            P[5, 0] = p_y**2
            
            w_ij = (R_j / R_i)**2 * calculate_weight(r_ij, R_e)
            M += w_ij * np.dot(P, P.T)
            b_temp[j] = w_ij * P
        M_inv = np.linalg.inv(M)
        MinvHrs = np.dot(H_rs, M_inv)
        
        for j in range(n_neighbor[i]):
            idx_j = neighbor_idx[j]
            #i[indexdx_i].append(idx_j)
            Eta = np.dot(MinvHrs, b_temp[j])
            #print(Eta)
            EtaDx[k,idx_j] = Eta[1,0]
            EtaDy[k,idx_j] = Eta[2,0]
            EtaDxx[k,idx_j] = Eta[3,0]
            EtaDxy[k,idx_j] = Eta[4,0]
            EtaDyy[k,idx_j] = Eta[5,0]
            
        k += 1
            
    return EtaDx, EtaDy, EtaDxx, EtaDxy, EtaDyy

def LSMPSbUpwind(node_x, node_y, index, Rmax, R, r_e, R_s, neighbor, n_neighbor, fx, fy):
    N = len(node_x)
    n = len(index)
    EtaDx   = sparse.lil_matrix((n,N), dtype=np.float64)
    EtaDy   = sparse.lil_matrix((n,N), dtype=np.float64)
    EtaDxx  = sparse.lil_matrix((n,N), dtype=np.float64)
    EtaDxy  = sparse.lil_matrix((n,N), dtype=np.float64)
    EtaDyy  = sparse.lil_matrix((n,N), dtype=np.float64)
    
    k = 0
    for i in index:
        H_rs = np.zeros((6,6), dtype=np.float64)
        M = np.zeros((6,6), dtype=np.float64)
        P = np.zeros((6,1), dtype=np.float64)
        b_temp = np.zeros((n_neighbor[i], 6, 1), dtype=np.float64)
        
       # print('Calculating derivative for particle ' + str(i) + '/' + str(N))
        
        neighbor_idx = neighbor[i]
        
        #R_max = np.max(R[neighbor_idx])
        
        idx_i = i
        x_i = node_x[idx_i]
        y_i = node_y[idx_i]
        R_e = r_e * Rmax[idx_i]
        R_i = R[idx_i]
                
        H_rs[0, 0] = 1
        H_rs[1, 1] = R_s[i]**-1
        H_rs[2, 2] = R_s[i]**-1
        H_rs[3, 3] = 2 * R_s[i]**-2
        H_rs[4, 4] = R_s[i]**-2
        H_rs[5, 5] = 2 * R_s[i]**-2
                
        for j in range(n_neighbor[i]):
            idx_j = neighbor_idx[j]
            x_j = node_x[idx_j]
            y_j = node_y[idx_j]
            R_j = R[idx_j]
            
            x_ij = x_j - x_i
            y_ij = y_j - y_i
            r_ij = np.sqrt((x_ij)**2 + (y_ij)**2)
             
            fx_i = fx[i]
            fy_i = fy[i]
            
            r_ij = np.sqrt((x_ij)**2 + (y_ij)**2)
            
            if r_ij <= 1e-12:
                n_ij = np.array([0.0,0.0])
            else:
                n_ij = np.array([x_ij, y_ij]) / r_ij
            if fx_i <= 1e-12:
                if fy_i <= 1e-12:
                    n_upwind = np.array([0.0,0.0])
                else:
                    n_upwind = np.array([0.0,-fy_i/abs(fy_i)])
            elif fy_i <= 1e-12:
                if fx_i <=1e-12:
                    n_upwind = np.array([0.0,0.0])
                else:
                    n_upwind = np.array([-fx_i/abs(fx_i),0.0])
            else:
                n_upwind = np.array([-fx_i/abs(fx_i), -fy_i/abs(fy_i)])
            if n_ij[0] * n_upwind[0] +  n_ij[1] * n_upwind[1] >= -1e-12:
                w_ij = (R_j / R_i)**2 * calculate_weight(r_ij, R_e)
            else:
                w_ij = 1e-12
             
            p_x = x_ij / R_s[i]
            p_y = y_ij / R_s[i]
            
            P[0, 0] = 1.0
            P[1, 0] = p_x
            P[2, 0] = p_y
            P[3, 0] = p_x**2
            P[4, 0] = p_x * p_y
            P[5, 0] = p_y**2
            
            M += w_ij * np.dot(P, P.T)
            b_temp[j] = w_ij * P
        M_inv = np.linalg.inv(M)
        MinvHrs = np.dot(H_rs, M_inv)
        
        for j in range(n_neighbor[i]):
            idx_j = neighbor_idx[j]
            #i[indexdx_i].append(idx_j)
            Eta = np.dot(MinvHrs, b_temp[j])
            #print(Eta)
            EtaDx[k,idx_j] = Eta[1,0]
            EtaDy[k,idx_j] = Eta[2,0]
            EtaDxx[k,idx_j] = Eta[3,0]
            EtaDxy[k,idx_j] = Eta[4,0]
            EtaDyy[k,idx_j] = Eta[5,0]
            
        k += 1
            
    return EtaDx, EtaDy, EtaDxx, EtaDxy, EtaDyy
#%%

RAD = 0.5
xcenter = 1 * (2 * RAD)
ycenter = 1 * (2 * RAD)
xmin = 0
xmax = xcenter + 2 * (2 * RAD)
ymin = 0
ymax = ycenter + 1 * (2 * RAD)
sigma = 0.02
r_e = 2.1
r_s = 1.0
sphere = True
# %%
node_x, node_y, node_z, normal_x_bound, normal_y_bound, n_boundary, index, diameter = generate_particles_singleres(xmin, xmax, ymin, ymax, sigma, RAD)
n_particle = node_x.shape[0]
cell_size = r_e * np.max(diameter)
# %%
# Neighbor search
n_bound = n_boundary[3]
h = np.ones_like(diameter) * np.max(diameter)
rc = np.concatenate((h[:n_bound] * r_e, h[n_bound:] * r_e))
nodes_3d = np.concatenate((node_x.reshape(-1,1), node_y.reshape(-1,1), node_z.reshape(-1,1)), axis = 1)
neighbor, n_neighbor = multiple_verlet(nodes_3d, n_bound, rc)
#%%
Rmax = np.zeros(n_particle)
for i in range(n_particle):
    ni = np.array(neighbor[i])
    Rmax[i] = np.array([np.max(diameter[ni])])
    
R_s = r_s * diameter
R = diameter / 2

index = np.array(index)
i_list = np.array_split(index, n_thread)
res = Parallel(n_jobs=-1)(delayed(LSMPSb)(node_x, node_y, idx, Rmax, R, r_e, R_s, neighbor, n_neighbor) for idx in i_list)
#%%
EtaDx = [0] * n_thread
EtaDy = [0] * n_thread
EtaDxx = [0] * n_thread
EtaDxy = [0] * n_thread
EtaDyy = [0] * n_thread

for i in range(n_thread):
   EtaDx[i] = res[i][0]
   EtaDy[i] = res[i][1]
   EtaDxx[i] = res[i][2]
   EtaDxy[i] = res[i][3]
   EtaDyy[i] = res[i][4]

EtaDx = sparse.csr_matrix(sparse.vstack(EtaDx))
EtaDy = sparse.csr_matrix(sparse.vstack(EtaDy))
EtaDxx = sparse.csr_matrix(sparse.vstack(EtaDxx))
EtaDxy = sparse.csr_matrix(sparse.vstack(EtaDxy))
EtaDyy = sparse.csr_matrix(sparse.vstack(EtaDyy))
   
#%%
#! INITIAL condition
# a uniform and divergence free field if possible, default is zero
V0_2d = np.zeros((n_particle,3))
u = np.zeros(n_particle) # Initialize rhs vector
v = np.zeros(n_particle) # Initialize rhs vector
p = np.zeros(n_particle) # Initialize rhs vectory

#! BOUNDARY CONDITION
# Both for velocity and pressure
# 1. LHS MATRIX corresponding to the boundary particles
# 2. RHS VECTOR corresponding to the boundary particles

# Initialize LHS boundary operators
I_2d = sparse.identity(n_particle).tocsr() # Identity
#* LHS:: default boundary operator for velocity (& streamfcn) is Dirichlet
u_bound_2d = I_2d[:n_bound]
v_bound_2d = u_bound_2d.copy()
# psi_bound_2d = u_bound_2d.copy()
#* LHS:: default boundary operator for pressure is Neumann
nbx = normal_x_bound[:n_bound].reshape(n_bound,1)
nby = normal_y_bound[:n_bound].reshape(n_bound,1)
p_bound_2d = EtaDx[:n_bound].multiply(nbx) \
            + EtaDy[:n_bound].multiply(nby) 

#* RHS:: default value is 0
rhs_u = np.zeros(n_bound) # Initialize rhs vector
rhs_v = np.zeros(n_bound) # Initialize rhs vector
rhs_p = np.zeros(n_bound) # Initialize rhs vectory

#%%
# east
idx_begin   = 0
idx_end     = n_boundary[0]   
if sphere:
    u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end] #* neumann velocity
    v_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end] #* neumann velocity
    p_bound_2d[idx_begin:idx_end] = I_2d[idx_begin:idx_end] # * dirichlet pressure
# p_bound_2d[idx_begin] = I_2d[idx_begin]
#* rhs
# rhs_u_[idx_begin:idx_end] = 1.0

#? west
idx_begin   = idx_end
idx_end     = n_boundary[1]
# u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
#* rhs
if sphere:
    rhs_u[idx_begin:idx_end] = 1.0
    u[idx_begin:idx_end] = 1.0

#? north
idx_begin   = idx_end
idx_end     = n_boundary[2]
#* rhs
if sphere:
    rhs_u[idx_begin:idx_end] = 1.0
    u[idx_begin:idx_end] = 1.0
#? south
idx_begin   = idx_end
idx_end     = n_boundary[3]
# u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
#* rhs
if sphere:
    rhs_u[idx_begin:idx_end] = 1.0
    u[idx_begin:idx_end] = 1.0
#%%
n_thread = threading.active_count()
i_list = np.array_split(index, n_thread)
resUpwind = Parallel(n_jobs=-1)(delayed(LSMPSbUpwind)(node_x, node_y, idx, Rmax, R, r_e, R_s, neighbor, n_neighbor, u, v) for idx in i_list)
EtaDxUpwind = [0] * n_thread
EtaDyUpwind = [0] * n_thread
EtaDxxUpwind = [0] * n_thread
EtaDxyUpwind = [0] * n_thread
EtaDyyUpwind = [0] * n_thread

for i in range(n_thread):
   EtaDxUpwind[i] = resUpwind[i][0]
   EtaDyUpwind[i] = resUpwind[i][1]
   EtaDxxUpwind[i] = resUpwind[i][2]
   EtaDxyUpwind[i] = resUpwind[i][3]
   EtaDyyUpwind[i] = resUpwind[i][4]

EtaDxUpwind = sparse.csr_matrix(sparse.vstack(EtaDxUpwind))
EtaDyUpwind = sparse.csr_matrix(sparse.vstack(EtaDyUpwind))
EtaDxxUpwind = sparse.csr_matrix(sparse.vstack(EtaDxxUpwind))
EtaDxyUpwind = sparse.csr_matrix(sparse.vstack(EtaDxyUpwind))
EtaDyyUpwind = sparse.csr_matrix(sparse.vstack(EtaDyyUpwind))
#%%
# Poisson matrix
idx_begin = n_boundary[3]
idx_end = n_particle
# create Poisson matrix
poisson_2d = EtaDxx + EtaDyy
poisson_2d = sparse.vstack((p_bound_2d, poisson_2d[idx_begin:idx_end]))
poisson_2d = sparse.linalg.factorized(poisson_2d.tocsc())
#%%
# NS solver
alphaC = 0.1
#dt = np.min(alphaC * diameter / np.sqrt(u**2+v**2))
dt = 0.005
#dt = 0.05
nu = 0.01
eta = 1e-4
T = 0

idx_begin = n_boundary[3]
idx_end = n_particle

dx_2d = EtaDx
dy_2d = EtaDy
dxx_2d = EtaDxx
dxy_2d = EtaDxy
dyy_2d = EtaDyy

dx_upwind = EtaDxUpwind
dy_upwind = EtaDyUpwind

#%%
# Solving initial pressure field
n_inner = idx_end - idx_begin

u_conv = u.reshape(n_particle,1)
v_conv = v.reshape(n_particle,1)

conv_2d = dx_upwind.multiply(u_conv) + dy_upwind.multiply(v_conv)

RHS_p = -dx_2d[n_bound:].dot(conv_2d.dot(u)) - dy_2d[n_bound:].dot(conv_2d.dot(v))
RHS_p = np.concatenate((rhs_p, RHS_p))
p0 = poisson_2d(RHS_p)
#%%
in_solid_ = (node_x - xcenter)**2 + (node_y - ycenter)**2 <= (RAD+1e-13)**2 
darcy_drag_ = (1 / eta)*in_solid_.astype(float)
Ddrag_2d = I_2d.multiply(darcy_drag_.reshape(-1,1))
Ddrag_2d = sparse.csr_matrix(Ddrag_2d)
#%%
# Solving predicted velocity
diff_2d = I_2d[n_bound:] - nu * dt * (dxx_2d[n_bound:] + dyy_2d[n_bound:])
in_LHS_2d = diff_2d + conv_2d[n_bound:] + Ddrag_2d[n_bound:]
# solve for u
RHS_u = u[n_bound:] - dt * dx_2d[n_bound:].dot(p0)
RHS_u = np.concatenate((rhs_u, RHS_u))
LHS_2d = sparse.vstack((u_bound_2d, in_LHS_2d))

u_pred = linalg.spsolve(LHS_2d, RHS_u)

# solve for v
RHS_v = v[n_bound:] - dt * dy_2d[n_bound:].dot(p0)
RHS_v = np.concatenate((rhs_v, RHS_v))
LHS_2d = sparse.vstack((v_bound_2d, in_LHS_2d))

v_pred = linalg.spsolve(LHS_2d, RHS_v)
#%%
# Solve p1
ddrag_u = Ddrag_2d.dot(u_pred - 0)
ddrag_v = Ddrag_2d.dot(v_pred - 0)
RHS_p = (dxx_2d[n_bound:] + dyy_2d[n_bound:]).dot(p0) \
        + 1 / dt * (dx_2d[n_bound:].dot(u_pred) + dy_2d[n_bound:].dot(v_pred)) \
        + dx_2d[n_bound:].dot(ddrag_u) + dy_2d[n_bound:].dot(ddrag_v)
RHS_p = np.concatenate((rhs_p, RHS_p))

p1 = poisson_2d(RHS_p)
#%%
# Solving corrected velocity
in_LHS_2d = I_2d[n_bound:] + dt * Ddrag_2d[n_bound:]

# solve for u
RHS_u = u_pred[n_bound:] - dt * (dx_2d[n_bound:].dot(p1 - p0) + Ddrag_2d[n_bound:].dot(u_pred))
RHS_u = np.concatenate((rhs_u, RHS_u))
LHS_2d = sparse.vstack((u_bound_2d, in_LHS_2d))

u_corr = linalg.spsolve(LHS_2d, RHS_u)

# solve for v
RHS_v = v_pred[n_bound:] - dt * (dy_2d[n_bound:].dot(p1 - p0) + Ddrag_2d[n_bound:].dot(v_pred))
RHS_v = np.concatenate((rhs_v, RHS_v))

LHS_2d = sparse.vstack((v_bound_2d, in_LHS_2d))

v_corr = linalg.spsolve(LHS_2d, RHS_v)

u1, v1 = u_corr, v_corr
u0, v0 = u, v
#%%
diff_2d = nu * (EtaDxx + EtaDyy)
T += dt
CL = []
CD = []
ts = []
#%%
while T < 5:
    #dt = np.min(alphaC * diameter / np.sqrt(u1**2+v1**2))
    print(' T = ', T)
    n_thread = threading.active_count()
    i_list = np.array_split(index, n_thread)
    resUpwind = Parallel(n_jobs=-1)(delayed(LSMPSbUpwind)(node_x, node_y, idx, Rmax, R, r_e, R_s, neighbor, n_neighbor, u1, v1) for idx in i_list)
    
    EtaDxUpwind = [0] * n_thread
    EtaDyUpwind = [0] * n_thread
    EtaDxxUpwind = [0] * n_thread
    EtaDxyUpwind = [0] * n_thread
    EtaDyyUpwind = [0] * n_thread

    for i in range(n_thread):
       EtaDxUpwind[i] = resUpwind[i][0]
       EtaDyUpwind[i] = resUpwind[i][1]

    EtaDxUpwind = sparse.csr_matrix(sparse.vstack(EtaDxUpwind))
    EtaDyUpwind = sparse.csr_matrix(sparse.vstack(EtaDyUpwind))
    
    dx_upwind = EtaDxUpwind
    dy_upwind = EtaDyUpwind
    
    # 1, Velocity prediction
    # Calculate predicted velocity
    # Create LHS matrix for velocity
    u_conv = u1.reshape(n_particle,1)
    v_conv = v1.reshape(n_particle,1)
    conv_2d = dx_upwind.multiply(u_conv) + dy_upwind.multiply(v_conv)
    diff_2d = I_2d[n_bound:] - 2 / 3 * dt * nu * (dxx_2d[n_bound:] + dyy_2d[n_bound:])
    in_LHS_2d = diff_2d + 2 / 3 * dt * (conv_2d[n_bound:] + Ddrag_2d[n_bound:])
    # solve for u
    LHS_2d = sparse.vstack((u_bound_2d, in_LHS_2d))
    RHS_u = 4 / 3 * u1[n_bound:] - 1 / 3 * u0[n_bound:] \
            - 2 / 3 * dt * dx_2d[n_bound:].dot(p1)
    RHS_u = np.concatenate((rhs_u, RHS_u))
    
    u_pred = linalg.spsolve(LHS_2d, RHS_u)
    # solve for v
    LHS_2d = sparse.vstack((v_bound_2d, in_LHS_2d))
    RHS_v = 4 / 3 * v1[n_bound:] - 1 / 3 * v0[n_bound:] \
            - 2 / 3 * dt * dy_2d[n_bound:].dot(p1)
    RHS_v = np.concatenate((rhs_v, RHS_v))
    
    v_pred = linalg.spsolve(LHS_2d, RHS_v)
    
    # 2. Pressure correction
    # Calculate value for phi
    ddrag_u = Ddrag_2d.dot(u_pred - 0)
    ddrag_v = Ddrag_2d.dot(v_pred - 0)
    RHS_phi = 3 / (2 * dt) * (dx_2d[n_bound:].dot(u_pred) + dy_2d[n_bound:].dot(v_pred)) \
                + dx_2d[n_bound:].dot(ddrag_u) \
                + dy_2d[n_bound:].dot(ddrag_v)
                
    RHS_p = np.concatenate((rhs_p, RHS_phi))
    
    phi = poisson_2d(RHS_p)
    
    divV = dx_2d.dot(u_pred) + dy_2d.dot(v_pred)
    
    p = phi + p1 - nu * divV
    
    # Velocity correction
    # Create LHS matrix
    in_LHS_2d = I_2d[n_bound:] + Ddrag_2d[n_bound:]
    
    # solve for u
    LHS_2d = sparse.vstack((u_bound_2d, in_LHS_2d))
    RHS_u = u_pred[n_bound:] - 2 / 3 * dt * (dx_2d[n_bound:].dot(phi) + Ddrag_2d[n_bound:].dot(u_pred))
    RHS_u = np.concatenate((rhs_u, RHS_u))
    
    u_corr = linalg.spsolve(LHS_2d, RHS_u)
    
    # solve for v
    LHS_2d = sparse.vstack((v_bound_2d, in_LHS_2d))
    RHS_v = v_pred[n_bound:] - 2 / 3 * dt * (dy_2d[n_bound:].dot(phi) + Ddrag_2d[n_bound:].dot(v_pred))
    RHS_v = np.concatenate((rhs_v, RHS_v))
    
    v_corr = linalg.spsolve(LHS_2d, RHS_v)
    
    u0, v0 = u1, v1
    u1, v1 = u_corr, v_corr
    p1 = p
    
    print(' max vres = ', np.max(np.sqrt(u_pred**2+v_pred**2)))
    # Force Calculation
    u_solid = (u_pred[in_solid_] - 0) / eta
    v_solid = (v_pred[in_solid_] - 0) / eta
    h_solid = diameter[in_solid_]
    c_x = np.sum((u_solid) *(h_solid**2)) / (0.5 * np.pi * RAD**2)
    c_y = np.sum((v_solid) *(h_solid**2)) / (0.5 * np.pi * RAD**2)
    CD.append(c_x)
    CL.append(c_y)
    ts.append(T)
    print(' CD = ', c_x)
    print(' CL = ', c_y)
    print(' dt = ', dt)
    
    T += dt   
#%%
visualize(node_x, node_y, p1, diameter, 'initial_pressure.png')
visualize(node_x, node_y, u1, diameter, 'initial_pressure.png')
visualize(node_x, node_y, v1, diameter, 'initial_pressure.png')
