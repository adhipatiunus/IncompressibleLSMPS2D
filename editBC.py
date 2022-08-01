#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:40:12 2022
@author: adhipatiunus
"""
# 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
import numpy as np
import scipy.sparse as sc
import scipy.sparse.linalg as scl
import matplotlib.pyplot as plt

from generate_particle import generate_particles, generate_particles_singleres
from neighbor_search import neighbor_search_cell_list
from LSMPS import LSMPSb, LSMPSbUpwind
from visualize import visualize

xmin = 0
xmax = 2
ymin = 0
ymax = 2
xcenter = 1
ycenter = 1
sigma = 0.01
r_e = 2.1
r_s = 1.3
RAD = 0.2
# %%
node_x, node_y, normal_x_bound, normal_y_bound, n_boundary, index, diameter = generate_particles_singleres(xmin, xmax, ymin, ymax, sigma, RAD)
cell_size = r_e * np.max(diameter)
neighbor = neighbor_search_cell_list(node_x, node_y, index, cell_size, ymax, ymin, xmax, xmin)
R_e = r_e * sigma
R_s = r_s * sigma
# %%
alphaC = 0.01
nu = 0.1
eta = 1e-4
etaP = 1e-4

n_particle = len(node_x)

p_type = np.array(['inner'] * n_particle)
u_type = np.array(['inner'] * n_particle)
v_type = np.array(['inner'] * n_particle)
w_type = np.array(['inner'] * n_particle)

# Initial Condition
u = np.zeros(n_particle)
v = np.zeros(n_particle)
w = np.zeros(n_particle)
dudn = np.zeros(n_particle)
dvdn = np.zeros(n_particle)
dwdn = np.zeros(n_particle)

p = np.zeros(n_particle)

# Boundary Condition
# East
idx_begin = 0
idx_end = n_boundary[0]
p[idx_begin:idx_end] = 0.0
# West
idx_begin = n_boundary[0]
idx_end = n_boundary[1]
u[idx_begin:idx_end] = 1.0
# North
idx_begin = n_boundary[1]
idx_end = n_boundary[2]
u[idx_begin:idx_end] = 1.0
# South
idx_begin = n_boundary[2]
idx_end = n_boundary[3]
u[idx_begin:idx_end] = 1.0
#%%
T = 0
dt = np.min(alphaC * diameter / np.sqrt(u**2+v**2))

EtaDx, EtaDy, EtaDxx, EtaDxy, EtaDyy = LSMPSb(
    node_x, node_y, index, R_e, R_s, neighbor)
EtaDxU, EtaDyU, EtaDxxU, EtaDxyU, EtaDyyU = LSMPSbUpwind(
    node_x, node_y, index, R_e, R_s, neighbor, u, v)
# %%
# Calculate Darcy drag
I_2D = np.eye(n_particle)
I_inner_2d = I_2D
in_solid_ = (node_x - xcenter)**2 + (node_y - ycenter)**2 <= (RAD+1e-13)**2
darcy_drag_ = (1/eta)*in_solid_.astype(float)
darcy_drag_p = (1/etaP)*in_solid_.astype(float)
Ddrag_2d = I_inner_2d * (darcy_drag_.reshape(-1, 1))
Ddrag_2d_p = I_inner_2d * (darcy_drag_p.reshape(-1, 1))
u_obs = np.zeros(n_particle)
v_obs = np.zeros(n_particle)
# Calculate initial pressure field from given velocity field (p0)
# Create LHS Matrix
LHS = np.zeros((n_particle, n_particle))
idx_begin = 0
idx_end = n_boundary[0]
LHS[idx_begin:idx_end, idx_begin:idx_end] = np.eye(idx_end - idx_begin)
idx_begin = idx_end
idx_end = n_boundary[3]
LHS[idx_begin:idx_end] = (EtaDx[idx_begin:idx_end].T * normal_x_bound[idx_begin:idx_end]).T \
    + (EtaDy[idx_begin:idx_end].T * normal_y_bound[idx_begin:idx_end]).T
idx_begin = idx_end
idx_end = n_particle
LHS[idx_begin:idx_end] = EtaDxx[idx_begin:idx_end] + EtaDyy[idx_begin:idx_end]

# Create RHS Vector
RHS = np.zeros(n_particle)
idx_begin = 0
idx_end = n_boundary[0]
RHS[idx_begin:idx_end] = 0
idx_begin = idx_end
idx_end = n_boundary[3]
RHS[idx_begin:idx_end] = 0
idx_begin = idx_end
idx_end = n_particle
conv_2d = (EtaDxU.T * u).T + (EtaDyU.T * v).T
RHS[idx_begin:idx_end] = - EtaDx[idx_begin:idx_end] @ (conv_2d @ u) \
                            - EtaDy[idx_begin:idx_end] @ (conv_2d @ v)
p0 = np.linalg.solve(LHS, RHS)
# %%
# Calculate predicted velocity
# Create LHS Matrix
LHS = np.zeros((n_particle, n_particle))
idx_begin = 0
idx_end = n_boundary[0]
LHS[idx_begin:idx_end] = (EtaDx[idx_begin:idx_end].T * normal_x_bound[idx_begin:idx_end]).T \
    + (EtaDy[idx_begin:idx_end].T * normal_y_bound[idx_begin:idx_end]).T
idx_begin = idx_end
idx_end = n_boundary[3]
LHS[idx_begin:idx_end, idx_begin:idx_end] = np.eye(idx_end - idx_begin)
idx_begin = idx_end
idx_end = n_particle
conv_2d = (EtaDxU.T * u).T + (EtaDyU.T * v).T
diff_2d = nu * (EtaDxx + EtaDyy)
LHS[idx_begin:idx_end, idx_begin:idx_end] = np.eye(idx_end - idx_begin) / dt
LHS[idx_begin:idx_end] += conv_2d[idx_begin:idx_end] - \
    diff_2d[idx_begin:idx_end] + Ddrag_2d[idx_begin:idx_end]
# Create RHS vector
# Solve for u
RHS = np.zeros(n_particle)
idx_begin = 0
idx_end = n_boundary[0]
RHS[idx_begin:idx_end] = 0
idx_begin = idx_end
idx_end = n_boundary[3]
RHS[idx_begin:idx_end] = u[idx_begin:idx_end]
idx_begin = idx_end
idx_end = n_particle
RHS[idx_begin:idx_end] = u[idx_begin:idx_end] / dt \
    - EtaDx[idx_begin:idx_end] @ p0
u_pred = np.linalg.solve(LHS, RHS)
# Solve for v
RHS = np.zeros(n_particle)
idx_begin = 0
idx_end = n_boundary[0]
RHS[idx_begin:idx_end] = 0
idx_begin = idx_end
idx_end = n_boundary[3]
RHS[idx_begin:idx_end] = v[idx_begin:idx_end]
idx_begin = idx_end
idx_end = n_particle
RHS[idx_begin:idx_end] = v[idx_begin:idx_end] / dt \
    - EtaDy[idx_begin:idx_end] @ p0
v_pred = np.linalg.solve(LHS, RHS)

# %%
conv_2d = (EtaDxU.T * u_pred).T + (EtaDyU.T * v_pred).T
LHS = np.zeros((n_particle, n_particle))
idx_begin = 0
idx_end = n_boundary[0]
LHS[idx_begin:idx_end, idx_begin:idx_end] = np.eye(idx_end - idx_begin)
idx_begin = idx_end
idx_end = n_boundary[3]
LHS[idx_begin:idx_end] = (EtaDx[idx_begin:idx_end].T * normal_x_bound[idx_begin:idx_end]).T \
    + (EtaDy[idx_begin:idx_end].T * normal_y_bound[idx_begin:idx_end]).T
idx_begin = idx_end
idx_end = n_particle
LHS[idx_begin:idx_end] = EtaDxx[idx_begin:idx_end] + EtaDyy[idx_begin:idx_end]

# Create RHS Vector
RHS = np.zeros(n_particle)
idx_begin = 0
idx_end = n_boundary[0]
RHS[idx_begin:idx_end] = 0
idx_begin = idx_end
idx_end = n_boundary[3]
RHS[idx_begin:idx_end] = 0
idx_begin = idx_end
idx_end = n_particle
darcy_u = Ddrag_2d_p @ (u_pred - u_obs)
darcy_v = Ddrag_2d_p @ (v_pred - v_obs)
RHS[idx_begin:idx_end] = (EtaDxx[idx_begin:idx_end] + EtaDyy[idx_begin:idx_end]) @ p0 \
    + 1 / dt * (EtaDx[idx_begin:idx_end] @ u_pred + EtaDy[idx_begin:idx_end] @ v_pred) \
    + EtaDx[idx_begin:idx_end] @ darcy_u \
    + EtaDy[idx_begin:idx_end] @ darcy_v
p1 = np.linalg.solve(LHS, RHS)
# %%
# Solve corrected velocity
u_corr = u_pred.copy()
v_corr = v_pred.copy()
idx_begin = 0
idx_end = n_boundary[0]
u_corr[idx_begin:idx_end] = u_pred[idx_begin:idx_end] - \
    dt * (EtaDx[idx_begin:idx_end] @ (p1 - p0))
v_corr[idx_begin:idx_end] = v_pred[idx_begin:idx_end] - \
    dt * (EtaDy[idx_begin:idx_end] @ (p1 - p0))
idx_begin = n_boundary[3]
idx_end = n_particle
u_corr[idx_begin:idx_end] = u_pred[idx_begin:idx_end] - \
    dt * (EtaDx[idx_begin:idx_end] @ (p1 - p0))
v_corr[idx_begin:idx_end] = v_pred[idx_begin:idx_end] - \
    dt * (EtaDy[idx_begin:idx_end] @ (p1 - p0))
u1, v1 = u_corr, v_corr
u0, v0 = u, v
# %%
diff_2d = nu * (EtaDxx + EtaDyy)
T += dt
CL = []
CD = []
ts = []
while T < 100 * dt:
    dt = np.min(alphaC * diameter / np.sqrt(u1**2+v1**2))
    #print(dt)
    # 1, Velocity prediction
    # Calculate predicted velocity
    # Create LHS matrix
    EtaDxU, EtaDyU, EtaDxxU, EtaDxyU, EtaDyyU = LSMPSbUpwind(
        node_x, node_y, index, R_e, R_s, neighbor, u1, v1)
    conv_2d = (EtaDxU.T * u1).T + (EtaDyU.T * v1).T
    LHS = np.zeros((n_particle, n_particle))
    idx_begin = 0
    idx_end = n_boundary[0]
    LHS[idx_begin:idx_end] = (EtaDx[idx_begin:idx_end].T * normal_x_bound[idx_begin:idx_end]).T \
        + (EtaDy[idx_begin:idx_end].T * normal_y_bound[idx_begin:idx_end]).T
    idx_begin = idx_end
    idx_end = n_boundary[3]
    LHS[idx_begin:idx_end, idx_begin:idx_end] = np.eye(idx_end - idx_begin)
    idx_begin = idx_end
    idx_end = n_particle
    LHS[idx_begin:idx_end, idx_begin:idx_end] = np.eye(
        idx_end - idx_begin) * 3 / (2 * dt)
    LHS[idx_begin:idx_end] += conv_2d[idx_begin:idx_end] \
        - diff_2d[idx_begin:idx_end] \
        + Ddrag_2d[idx_begin:idx_end]
    # Create RHS vector
    # Solve for u
    RHS = np.zeros(n_particle)
    idx_begin = 0
    idx_end = n_boundary[0]
    RHS[idx_begin:idx_end] = 0
    idx_begin = idx_end
    idx_end = n_boundary[3]
    RHS[idx_begin:idx_end] = 1.0
    idx_begin = idx_end
    idx_end = n_particle
    RHS[idx_begin:idx_end] = (4 * u1[idx_begin:idx_end] - u0[idx_begin:idx_end]) / (2 * dt) \
        - EtaDx[idx_begin:idx_end] @ p1
    u_pred = np.linalg.solve(LHS, RHS)
    # Solve for v
    RHS = np.zeros(n_particle)
    idx_begin = 0
    idx_end = n_boundary[0]
    RHS[idx_begin:idx_end] = 0.0
    idx_begin = idx_end
    idx_end = n_boundary[3]
    RHS[idx_begin:idx_end] = 0.0
    idx_begin = idx_end
    idx_end = n_particle
    RHS[idx_begin:idx_end] = (4 * v1[idx_begin:idx_end] - v0[idx_begin:idx_end]) / (2 * dt) \
        - EtaDy[idx_begin:idx_end] @ p1
    v_pred = np.linalg.solve(LHS, RHS)

    # 2. Pressure correction
    # Solve Poisson equation
    # Create LHS matrix for phi
    LHS = np.zeros((n_particle, n_particle))
    idx_begin = 0
    idx_end = n_boundary[0]
    LHS[idx_begin:idx_end, idx_begin:idx_end] = np.eye(idx_end - idx_begin)
    idx_begin = idx_end
    idx_end = n_boundary[3]
    LHS[idx_begin:idx_end] = (EtaDx[idx_begin:idx_end].T * normal_x_bound[idx_begin:idx_end]).T \
        + (EtaDy[idx_begin:idx_end].T * normal_y_bound[idx_begin:idx_end]).T
    idx_begin = idx_end
    idx_end = n_particle
    LHS[idx_begin:idx_end] = EtaDxx[idx_begin:idx_end] + \
                                EtaDyy[idx_begin:idx_end]
    # Create RHS vector for phi
    RHS = np.zeros(n_particle)
    idx_begin = 0
    idx_end = n_boundary[0]
    RHS[idx_begin:idx_end] = 0
    idx_begin = idx_end
    idx_end = n_boundary[3]
    RHS[idx_begin:idx_end] = 0
    idx_begin = idx_end
    idx_end = n_particle
    darcy_u = Ddrag_2d_p @ (u_pred - u_obs)
    darcy_v = Ddrag_2d_p @ (v_pred - v_obs)
    RHS[idx_begin:idx_end] = 3 / (2 * dt) * (EtaDx[idx_begin:idx_end] @ u_pred + EtaDy[idx_begin:idx_end] @ v_pred) \
                            + EtaDx[idx_begin:idx_end] @ darcy_u \
                            + EtaDy[idx_begin:idx_end] @ darcy_v
    phi = np.linalg.solve(LHS, RHS)
    div = EtaDx @ u_pred + EtaDy @ v_pred
    p = p1.copy()
    idx_begin = n_boundary[0]
    idx_end = n_particle
    p[idx_begin:idx_end] = phi[idx_begin:idx_end] + \
                            p1[idx_begin:idx_end] - nu * div[idx_begin:idx_end]

    u_corr = u_pred.copy()
    v_corr = v_pred.copy()
    idx_begin = 0
    idx_end = n_boundary[0]
    u_corr[idx_begin:idx_end] = u_pred[idx_begin:idx_end] - \
                                2 * dt / 3 * (EtaDx[idx_begin:idx_end] @ phi)
    v_corr[idx_begin:idx_end] = v_pred[idx_begin:idx_end] - \
                                2 * dt / 3 * (EtaDy[idx_begin:idx_end] @ phi)
    idx_begin = n_boundary[3]
    idx_end = n_particle
    u_corr[idx_begin:idx_end] = u_pred[idx_begin:idx_end] - \
                                2 * dt / 3 * (EtaDx[idx_begin:idx_end] @ phi)
    v_corr[idx_begin:idx_end] = v_pred[idx_begin:idx_end] - \
                                2 * dt / 3 * (EtaDy[idx_begin:idx_end] @ phi)

    u0, v0 = u1, v1
    u1, v1 = u_corr, v_corr
    p1 = p
    print(' max vres = ', np.max(np.sqrt(u1**2+v1**2)))
    # Force Calculation
    u_solid = (u1[in_solid_] - u_obs[in_solid_]) / eta
    v_solid = (v1[in_solid_] - v_obs[in_solid_]) / eta
    h_solid = diameter[in_solid_]
    c_x = 2*(4/np.pi)*np.sum((u_solid) *(h_solid**2))
    c_y = 2*(4/np.pi)*np.sum((v_solid) *(h_solid**2))
    CD.append(c_x)
    CL.append(c_y)
    ts.append(T)
    print(' CD = ', c_x)
    print(' CL = ', c_y)
    print(' dt = ', dt)
    print(' T = ', T)
    
    #visualize(node_x, node_y, p1, diameter, 'initial_pressure.png')
    #visualize(node_x, node_y, np.sqrt(u_corr**2+v_corr**2), diameter, 'initial_pressure.png')
    #visualize(node_x, node_y, u_corr, diameter, 'initial_pressure.png')
    #visualize(node_x, node_y, v_corr, diameter, 'initial_pressure.png')

    T += dt

# %%
visualize(node_x, node_y, p1, diameter, 'initial_pressure.png')
visualize(node_x, node_y, np.sqrt(u_corr**2+v_corr**2), diameter, 'initial_pressure.png')
visualize(node_x, node_y, u_corr, diameter, 'initial_pressure.png')
visualize(node_x, node_y, v_corr, diameter, 'initial_pressure.png')
#plt.scatter(node_x, node_y, c=p0, cmap="jet", linewidth=0)
# %%
np.savez('x', node_x)
np.savez('y', node_y)
np.savez('u_corr', u_corr)
np.savez('v_corr', v_corr)
np.savez('u_pred', u_corr)
np.savez('v_pred', v_corr)
np.savez('p', p)
