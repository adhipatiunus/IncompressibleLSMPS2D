#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 19:27:54 2022

@author: adhipatiunus
"""
import numpy as np
import numba as nb
from scipy import sparse

@nb.njit
def calculate_weight(r_ij, R_e):
    if r_ij < R_e:
        w_ij = (1 - r_ij / R_e)**2
    else:
        w_ij = 0
    return w_ij

@nb.njit
def LSMPSb(node_x, node_y, index, Rmax, R, r_e, R_s, neighbor_matrix, n_neighbor):
    N = len(node_x)
    EtaDx   = np.zeros((N,N), dtype=np.float64)
    EtaDy   = np.zeros((N,N), dtype=np.float64)
    EtaDxx  = np.zeros((N,N), dtype=np.float64)
    EtaDxy  = np.zeros((N,N), dtype=np.float64)
    EtaDyy  = np.zeros((N,N), dtype=np.float64)

    for i in index:
        H_rs = np.zeros((6,6), dtype=np.float64)
        M = np.zeros((6,6), dtype=np.float64)
        P = np.zeros((6,1), dtype=np.float64)
        b_temp = np.zeros((n_neighbor[i], 6, 1), dtype=np.float64)
        
       # print('Calculating derivative for particle ' + str(i) + '/' + str(N))
        
        neighbor_idx = neighbor_matrix[i]
        
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
            EtaDx[idx_i,idx_j] = Eta[1,0]
            EtaDy[idx_i,idx_j] = Eta[2,0]
            EtaDxx[idx_i,idx_j] = Eta[3,0]
            EtaDxy[idx_i,idx_j] = Eta[4,0]
            EtaDyy[idx_i,idx_j] = Eta[5,0]
            
    return EtaDx, EtaDy, EtaDxx, EtaDxy, EtaDyy

@nb.njit
def LSMPSbUpwind(node_x, node_y, index, Rmax, R, r_e, R_s, neighbor_matrix, n_neighbor, fx, fy):
    N = len(node_x)
    EtaDx   = np.zeros((N,N), dtype=np.float64)
    EtaDy   = np.zeros((N,N), dtype=np.float64)
    EtaDxx  = np.zeros((N,N), dtype=np.float64)
    EtaDxy  = np.zeros((N,N), dtype=np.float64)
    EtaDyy  = np.zeros((N,N), dtype=np.float64)

    for i in index:
        H_rs = np.zeros((6,6), dtype=np.float64)
        M = np.zeros((6,6), dtype=np.float64)
        P = np.zeros((6,1), dtype=np.float64)
        b_temp = np.zeros((n_neighbor[i], 6, 1), dtype=np.float64)
        
       # print('Calculating derivative for particle ' + str(i) + '/' + str(N))
        
        neighbor_idx = neighbor_matrix[i]
        
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
            if n_ij[0] * n_upwind[0] +  n_ij[1] * n_upwind[1]> 0:
                w_ij = calculate_weight(r_ij, R_e)
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
            EtaDx[idx_i,idx_j] = Eta[1,0]
            EtaDy[idx_i,idx_j] = Eta[2,0]
            EtaDxx[idx_i,idx_j] = Eta[3,0]
            EtaDxy[idx_i,idx_j] = Eta[4,0]
            EtaDyy[idx_i,idx_j] = Eta[5,0]
            
    return EtaDx, EtaDy, EtaDxx, EtaDxy, EtaDyy
        
