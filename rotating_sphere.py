#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:42:13 2022

@author: adhipatiunus
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import threading
from joblib import Parallel, delayed

from generate_particle import generate_particles, generate_particles_rectangle, generate_particles_singleres
from neighbor_search import neighbor_search_cell_list
from neighbor_search_verlet import multiple_verlet
from visualize import visualize
#%%
def calculate_weight(r_ij, R_e):
    if r_ij < R_e:
        w_ij = (1 - r_ij / R_e)**2
    else:
        w_ij = 0
    return w_ij

def LSMPSb(node_x, node_y, index, diameter, r_e, neighbor, n_neighbor):
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
        
        neighbor_idx = np.array(neighbor[i])
        
        Li = np.mean(diameter[neighbor_idx])
        
        #R_max = np.max(R[neighbor_idx])
        
        idx_i = i
        x_i = node_x[idx_i]
        y_i = node_y[idx_i]
        R_i = r_e * Li
                
        H_rs[0, 0] = 1
        H_rs[1, 1] = Li**-1
        H_rs[2, 2] = Li**-1
        H_rs[3, 3] = 2 * Li**-2
        H_rs[4, 4] = Li**-2
        H_rs[5, 5] = 2 * Li**-2
                
        for j in range(n_neighbor[i]):
            idx_j = neighbor_idx[j]
            x_j = node_x[idx_j]
            y_j = node_y[idx_j]
            R_j = r_e * diameter[idx_j]
            R_ij = (R_i + R_j) / 2
            
            x_ij = x_j - x_i
            y_ij = y_j - y_i
            r_ij = np.sqrt((x_ij)**2 + (y_ij)**2)
             
            p_x = x_ij / Li
            p_y = y_ij / Li
            
            P[0, 0] = 1.0
            P[1, 0] = p_x
            P[2, 0] = p_y
            P[3, 0] = p_x**2
            P[4, 0] = p_x * p_y
            P[5, 0] = p_y**2
            
            w_ij = calculate_weight(r_ij, R_ij)
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

def LSMPSbUpwind(node_x, node_y, index, diameter, r_e, neighbor, n_neighbor, fx, fy):
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
        
        neighbor_idx = np.array(neighbor[i])
        
        Li = np.mean(diameter[neighbor_idx])
        
        #R_max = np.max(R[neighbor_idx])
        
        ignored_neighbor = []
        
        idx_i = i
        x_i = node_x[idx_i]
        y_i = node_y[idx_i]
        R_i = r_e * Li
                
        H_rs[0, 0] = 1
        H_rs[1, 1] = Li**-1
        H_rs[2, 2] = Li**-1
        H_rs[3, 3] = 2 * Li**-2
        H_rs[4, 4] = Li**-2
        H_rs[5, 5] = 2 * Li**-2
                
        for j in range(n_neighbor[i]):
            idx_j = neighbor_idx[j]
            x_j = node_x[idx_j]
            y_j = node_y[idx_j]
            R_j = r_e * diameter[idx_j]
            R_ij = (R_i + R_j) / 2
            
            x_ij = x_j - x_i
            y_ij = y_j - y_i
             
            p_x = x_ij / Li
            p_y = y_ij / Li
            
            P[0, 0] = 1.0
            P[1, 0] = p_x
            P[2, 0] = p_y
            P[3, 0] = p_x**2
            P[4, 0] = p_x * p_y
            P[5, 0] = p_y**2
            
            fx_i = fx[i]
            fy_i = fy[i]
            
            safety = 1e-12
            eps = 1e-3
            r_ij = np.sqrt((x_ij + safety)**2 + (y_ij + safety)**2)
            n_ij = np.array([x_ij + safety, y_ij + safety]) / (r_ij)
            n_upwind = -np.array([(fx_i + safety), (fy_i + safety)]) / (np.sqrt((fx_i + safety)**2 + (fy_i + safety)**2))
            w_spike = calculate_weight(r_ij, R_ij)
            if n_ij @ n_upwind > 1:
                theta_ij = 1
            elif n_ij @ n_upwind < -1:
                theta_ij = -1
            else:    
                theta_ij = np.arccos(n_ij @ n_upwind)
            w_ij = w_spike * max(np.cos(2 * theta_ij), eps)
            if np.isnan(w_ij):
                print(theta_ij)
            M += w_ij * np.dot(P, P.T)
            b_temp[j] = w_ij * P
        M_inv = np.linalg.inv(M)
        MinvHrs = np.dot(H_rs, M_inv)
        
        for j in range(n_neighbor[i]):
            idx_j = neighbor_idx[j]
            #i[indexdx_i].append(idx_j)
            Eta = np.dot(MinvHrs, b_temp[j])
            """
            if idx_j in ignored_neighbor:
                Eta = np.array([[0],[0],[0],[0],[0],[0]])
            """
            #print(Eta)
            EtaDx[k,idx_j] = Eta[1,0]
            EtaDy[k,idx_j] = Eta[2,0]
            EtaDxx[k,idx_j] = Eta[3,0]
            EtaDxy[k,idx_j] = Eta[4,0]
            EtaDyy[k,idx_j] = Eta[5,0]
            
        k += 1
    return EtaDx, EtaDy, EtaDxx, EtaDxy, EtaDyy
#%%

RAD = .5
xcenter = 1.5
ycenter = 2.5
xmin = 0
xmax = xcenter + 4.5
ymin = 0
ymax = ycenter + 2.5
width = 1
height = 1
sigma = 1e-2
r_e = 2.5
r_s = 1.0
brinkman = True
# %%
node_x, node_y, node_z, normal_x_bound, normal_y_bound, tangent_x_bound, tangent_y_bound, n_boundary, index, diameter = generate_particles_singleres(xmin, xmax, ymin, ymax, sigma, RAD)
n_particle = node_x.shape[0]
cell_size = r_e * np.max(diameter)
# %%
# Neighbor search
n_bound = n_boundary[3]
h = diameter 
rc = np.concatenate((h[:n_bound] * r_e, h[n_bound:] * r_e))
nodes_3d = np.concatenate((node_x.reshape(-1,1), node_y.reshape(-1,1), node_z.reshape(-1,1)), axis = 1)
neighbor, n_neighbor = multiple_verlet(nodes_3d, n_bound, rc)
#%%
n_thread = threading.active_count()
index = np.array(index)
i_list = np.array_split(index, n_thread)
res = Parallel(n_jobs=-1)(delayed(LSMPSb)(node_x, node_y, idx, diameter, r_e, neighbor, n_neighbor) for idx in i_list)

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
u0 = 1

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
#tbx = tangent_x_bound[:n_bound].reshape(n_bound,1)
#tby = tangent_y_bound[:n_bound].reshape(n_bound,1)
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
if brinkman:
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
if brinkman:
    rhs_u[idx_begin:idx_end] = u0
    u[idx_begin:idx_end] = u0

#? north
idx_begin   = idx_end
idx_end     = n_boundary[2]
#* rhs
if brinkman:
    rhs_u[idx_begin:idx_end] = u0
    u[idx_begin:idx_end] = u0
#? south
idx_begin   = idx_end
idx_end     = n_boundary[3]
# u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
#* rhs
if brinkman:
    rhs_u[idx_begin:idx_end] = u0
    u[idx_begin:idx_end] = u0
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
#dt = 0.0025
#dt = 0.05
Re = 2e1
nu = u0 * width / Re
eta = 1e-4
T = 0
omega = .2

idx_begin = n_boundary[3]
idx_end = n_particle

dx_2d = EtaDx
dy_2d = EtaDy
dxx_2d = EtaDxx
dxy_2d = EtaDxy
dyy_2d = EtaDyy
#%%
"""
safety = 1e-12
in_solid_ = (abs(node_x - xcenter) <= width / 2) * (abs(node_y - ycenter) <= height / 2)
darcy_drag_ = (1 / eta)*in_solid_.astype(float)
Ddrag_2d = I_2d.multiply(darcy_drag_.reshape(-1,1))
Ddrag_2d = sparse.csr_matrix(Ddrag_2d)
"""
in_solid_ = (node_x - xcenter)**2 + (node_y - ycenter)**2 <= (RAD+1e-13)**2 
darcy_drag_ = (1 / eta)*in_solid_.astype(float)
Ddrag_2d = I_2d.multiply(darcy_drag_.reshape(-1,1))
Ddrag_2d = sparse.csr_matrix(Ddrag_2d)
u_obs = -omega * (node_y - ycenter)
v_obs = omega * (node_x - xcenter)
convectionx_solid_ = 0
convectionz_solid_ = 0
u_obs = -omega * (node_y - ycenter)
v_obs = omega * (node_x - xcenter)
#%%
CL = []
CD = []
ts = []
#%%
dt = 5e-3
while T < 1e1:
    #dt = np.min(alphaC * diameter / np.sqrt(u**2+v**2))
    n_thread = threading.active_count()
    index = np.array(index)
    i_list = np.array_split(index, n_thread)
    res = Parallel(n_jobs=-1)(delayed(LSMPSbUpwind)(node_x, node_y, idx, diameter, r_e, neighbor, n_neighbor, u, v) for idx in i_list)

    EtaDxUpwind = [0] * n_thread
    EtaDyUpwind = [0] * n_thread

    for i in range(n_thread):
       EtaDxUpwind[i] = res[i][0]
       EtaDyUpwind[i] = res[i][1]
    
    EtaDxUpwind = sparse.csr_matrix(sparse.vstack(EtaDxUpwind))
    EtaDyUpwind = sparse.csr_matrix(sparse.vstack(EtaDyUpwind))
    
    dx_upwind = EtaDxUpwind
    dy_upwind = EtaDyUpwind
    
    #dt = 1e-3
    print(' T = ', T)
    # 1, Velocity prediction
    # Calculate predicted velocity
    # Create LHS matrix for velocity
    u_conv = u.reshape(n_particle,1)
    v_conv = v.reshape(n_particle,1)
    conv_2d = dx_upwind.multiply(u_conv) + dy_upwind.multiply(v_conv)
    diff_2d = I_2d[n_bound:] / dt - nu * (dxx_2d[n_bound:] + dyy_2d[n_bound:])
    in_LHS_2d = diff_2d + (conv_2d[n_bound:] + Ddrag_2d[n_bound:])
    # solve for u
    LHS_2d = sparse.vstack((u_bound_2d, in_LHS_2d))
    RHS_u = u[n_bound:] / dt + Ddrag_2d[n_bound:] @ u_obs
    RHS_u = np.concatenate((rhs_u, RHS_u))
    
    u_pred = linalg.spsolve(LHS_2d, RHS_u)
    # solve for v
    LHS_2d = sparse.vstack((v_bound_2d, in_LHS_2d))
    RHS_v = v[n_bound:] / dt + Ddrag_2d[n_bound:] @ v_obs
    RHS_v = np.concatenate((rhs_v, RHS_v))
    
    v_pred = linalg.spsolve(LHS_2d, RHS_v)
    
    # 2. Pressure correction
    # Calculate value for phi
    idx_begin = n_boundary[1]
    idx_end = n_boundary[3]
    
    u_bound = np.array([np.mean(u_pred[np.array(neighbor[i])]) for i in range(idx_begin, idx_end)])
    v_bound = np.array([np.mean(v_pred[np.array(neighbor[i])]) for i in range(idx_begin, idx_end)])
    
    
    rhs_p[idx_begin:idx_end] = 1 / dt * (u_bound * normal_x_bound[idx_begin:idx_end] + v_bound * normal_y_bound[idx_begin:idx_end])
    
    idx_begin = idx_end
    idx_end = n_particle
    
    RHS_phi = 1 / dt * (dx_2d[n_bound:].dot(u_pred) + dy_2d[n_bound:].dot(v_pred)) \
                + dx_2d[n_bound:].dot(Ddrag_2d.dot(u_pred)) + dy_2d[n_bound:].dot(Ddrag_2d.dot(v_pred))
     
    """
    RHS_phi = 1 / dt * (dx_2d[n_bound:].dot(u_pred) + dy_2d[n_bound:].dot(v_pred))
    """
    RHS_p = np.concatenate((rhs_p, RHS_phi))
    
    phi = poisson_2d(RHS_p)
    
    divV = dx_2d.dot(u_pred) + dy_2d.dot(v_pred)
    
    p = phi - nu * divV
    
    # Velocity correction
    # Create LHS matrix
    in_LHS_2d = I_2d[n_bound:] / dt + Ddrag_2d[n_bound:]
    
    # solve for u
    LHS_2d = sparse.vstack((u_bound_2d, in_LHS_2d))
    RHS_u = u_pred[n_bound:] / dt - (dx_2d[n_bound:].dot(phi) - Ddrag_2d[n_bound:].dot(u_pred))
    RHS_u = np.concatenate((rhs_u, RHS_u))
    
    u_corr = linalg.spsolve(LHS_2d, RHS_u)
    
    # solve for v
    LHS_2d = sparse.vstack((v_bound_2d, in_LHS_2d))
    RHS_v = v_pred[n_bound:] / dt - (dy_2d[n_bound:].dot(phi) - Ddrag_2d[n_bound:].dot(v_pred))
    RHS_v = np.concatenate((rhs_v, RHS_v))
    
    v_corr = linalg.spsolve(LHS_2d, RHS_v)
    
    u, v = u_corr, v_corr
    
    """
    u[n_bound:] = u_pred[n_bound:] - dt * dx_2d[n_bound:].dot(phi)
    v[n_bound:] = v_pred[n_bound:] - dt * dy_2d[n_bound:].dot(phi)
    """
    p1 = p
    
    print(' max vres = ', np.max(np.sqrt(u**2+v**2)))
    # Force Calculation
    u_solid = (u[in_solid_] - u_obs[in_solid_]) / eta
    v_solid = (v[in_solid_] - v_obs[in_solid_]) / eta
    h_solid = diameter[in_solid_]
    
    c_x = np.sum((u_solid) *(h_solid**2)) / (0.5 * (width))
    c_y = np.sum((v_solid) *(h_solid**2)) / (0.5 * (width))
    CD.append(c_x)
    CL.append(c_y)
    ts.append(T)
    print(' CD = ', c_x)
    print(' CL = ', c_y)
    print(' dt = ', dt)
    
    T += dt    
#%%
CD = np.array(CD)
CL = np.array(CL)
ts = np.array(ts)
np.save('x.npy', node_x)
np.save('y.npy', node_y)
np.save('u.npy', u)
np.save('v.npy', v)
np.save('p.npy', p)
np.save('CL.npy', CL)
np.save('CD.npy', CD)
np.save('ts.npy', ts)
#%%
import csv
vres = np.sqrt(u**2+v**2)
header = ['x', 'y', 'z','u', 'v', 'vres', 'p']
data = [[node_x[i], node_y[i], node_z[i], u[i], v[i], vres[i], p[i]] for i in range(len(node_x))]

with open('re100.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
