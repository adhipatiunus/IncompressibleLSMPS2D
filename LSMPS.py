#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 19:27:54 2022

@author: adhipatiunus
"""
import numpy as np

def calculate_weight(r_ij, R_e):
    if r_ij < R_e:
        w_ij = (1 - r_ij / R_e)**2
    else:
        w_ij = 0
    return w_ij

def LSMPSb(node_x, node_y, index, R_e, R_s, neighbor_list):
    N = len(node_x)
    b_data = [np.array([])] * N
    EtaDx   = np.zeros((N, N))
    EtaDy   = np.zeros((N, N))
    EtaDxx  = np.zeros((N, N))
    EtaDxy  = np.zeros((N, N))
    EtaDyy  = np.zeros((N, N))

    for i in index:
        H_rs = np.zeros((6,6))
        M = np.zeros((6,6))
        P = np.zeros((6,1))
        b_temp = [np.array([])] * len(neighbor_list[i])
        
       # print('Calculating derivative for particle ' + str(i) + '/' + str(N))
        
        neighbor_idx = np.array(neighbor_list[i])
        
        #R_max = np.max(R[neighbor_idx])
        
        idx_i = i
        x_i = node_x[idx_i]
        y_i = node_y[idx_i]
                
        H_rs[0, 0] = 1
        H_rs[1, 1] = R_s**-1
        H_rs[2, 2] = R_s**-1
        H_rs[3, 3] = 2 * R_s**-2
        H_rs[4, 4] = R_s**-2
        H_rs[5, 5] = 2 * R_s**-2
                
        for j in range(len(neighbor_idx)):
            idx_j = neighbor_idx[j]
            x_j = node_x[idx_j]
            y_j = node_y[idx_j]
            
            x_ij = x_j - x_i
            y_ij = y_j - y_i
            r_ij = np.sqrt((x_ij)**2 + (y_ij)**2)
             
            p_x = x_ij / R_s
            p_y = y_ij / R_s
            
            P[0, 0] = 1.0
            P[1, 0] = p_x
            P[2, 0] = p_y
            P[3, 0] = p_x**2
            P[4, 0] = p_x * p_y
            P[5, 0] = p_y**2
            
            w_ij = calculate_weight(r_ij, R_e)
            M += w_ij * np.matmul(P, P.T)
            b_temp[j] = w_ij * P
        M_inv = np.linalg.inv(M)
        MinvHrs = np.matmul(H_rs, M_inv)
        b_data[i] = b_temp
        
        for j in range(len(neighbor_idx)):
            idx_j = neighbor_idx[j]
            #i[indexdx_i].append(idx_j)
            Eta = np.matmul(MinvHrs, b_data[i][j])
            EtaDx[idx_i,idx_j] = Eta[1]
            EtaDy[idx_i,idx_j] = Eta[2]
            EtaDxx[idx_i,idx_j] = Eta[3]
            EtaDxy[idx_i,idx_j] = Eta[4]
            EtaDyy[idx_i,idx_j] = Eta[5]
            
    return EtaDx, EtaDy, EtaDxx, EtaDxy, EtaDyy

def LSMPSbUpwind(node_x, node_y, index, R_e, R_s, neighbor_list, fx, fy):
    N = len(node_x)
    b_data = [np.array([])] * N
    EtaDx   = np.zeros((N, N))
    EtaDy   = np.zeros((N, N))
    EtaDxx  = np.zeros((N, N))
    EtaDxy  = np.zeros((N, N))
    EtaDyy  = np.zeros((N, N))

    for i in index:
        H_rs = np.zeros((6,6))
        M = np.zeros((6,6))
        P = np.zeros((6,1))
        b_temp = [np.array([])] * len(neighbor_list[i])
        
        #print('Calculating derivative for particle ' + str(i) + '/' + str(N))
        
        neighbor_idx = np.array(neighbor_list[i])
        ignored_neighbor = []
        
        idx_i = i
        x_i = node_x[idx_i]
        y_i = node_y[idx_i]
        fx_i = fx[i]
        fy_i = fy[i]
                
        H_rs[0, 0] = 1
        H_rs[1, 1] = R_s**-1
        H_rs[2, 2] = R_s**-1
        H_rs[3, 3] = 2 * R_s**-2
        H_rs[4, 4] = R_s**-2
        H_rs[5, 5] = 2 * R_s**-2
                
        for j in range(len(neighbor_idx)):
            idx_j = neighbor_idx[j]
            x_j = node_x[idx_j]
            y_j = node_y[idx_j]
            
            x_ij = x_j - x_i
            y_ij = y_j - y_i
            r_ij = np.sqrt((x_ij)**2 + (y_ij)**2)
            if r_ij == 0:
                n_ij = np.array([0,0])
            else:
                n_ij = np.array([x_ij, y_ij]) / r_ij
            if fx_i == 0:
                if fy_i == 0:
                    n_upwind = np.array([0,0])
                else:
                    n_upwind = np.array([0,-fy_i/abs(fy_i)])
            elif fy_i == 0:
                if fx_i == 0:
                    n_upwind = np.array([0,0])
                else:
                    n_upwind = np.array([-fx_i/abs(fx_i),0])
            else:
                n_upwind = np.array([-fx_i/abs(fx_i), -fy_i/abs(fy_i)])
            if n_ij @ n_upwind > 0:
                w_ij = calculate_weight(r_ij, R_e)
            else:
                ignored_neighbor.append(idx_j)
                w_ij = 0
            #print(w_ij)
            p_x = x_ij / R_s
            p_y = y_ij / R_s
            
            P[0, 0] = 1.0
            P[1, 0] = p_x
            P[2, 0] = p_y
            P[3, 0] = p_x**2
            P[4, 0] = p_x * p_y
            P[5, 0] = p_y**2
            
            M += w_ij * np.matmul(P, P.T)
            b_temp[j] = w_ij * P
        if np.linalg.det(M) < 1e-6:
            """
            for j in ignored_neighbor:
                idx_j = j
                #print(idx_j)
                x_j = node_x[idx_j]
                y_j = node_y[idx_j]
                    
                x_ij = x_j - x_i
                y_ij = y_j - y_i
                r_ij = np.sqrt((x_ij)**2 + (y_ij)**2)
                    
                p_x = x_ij / R_s
                p_y = y_ij / R_s
                    
                P[0, 0] = 1.0
                P[1, 0] = p_x
                P[2, 0] = p_y
                P[3, 0] = p_x**2
                P[4, 0] = p_x * p_y
                P[5, 0] = p_y**2
                
                w_ij = calculate_weight(r_ij, R_e)
                    
                M += w_ij * np.matmul(P, P.T)
            """
            h = 0.005
            x = np.linspace(node_x[i] - h, node_x[i] + h, 3)
            y = np.linspace(node_y[i] - h, node_y[i] + h, 3)
            x_s, y_s = np.meshgrid(x, y)
            x_s = x_s.flatten()
            y_s = y_s.flatten()
            for j in range(len(x_s)):
                idx_j = j
                #print(idx_j)
                x_j = x_s[idx_j]
                y_j = y_s[idx_j]
                    
                x_ij = x_j - x_i
                y_ij = y_j - y_i
                r_ij = np.sqrt((x_ij)**2 + (y_ij)**2)
                    
                p_x = x_ij / R_s
                p_y = y_ij / R_s
                    
                P[0, 0] = 1.0
                P[1, 0] = p_x
                P[2, 0] = p_y
                P[3, 0] = p_x**2
                P[4, 0] = p_x * p_y
                P[5, 0] = p_y**2
                
                w_ij = calculate_weight(r_ij, R_e)
                    
                M += w_ij * np.matmul(P, P.T)
        #print(M)
        M_inv = np.linalg.inv(M)
        MinvHrs = np.matmul(H_rs, M_inv)
        b_data[i] = b_temp
        #print(b_temp)
        
        for j in range(len(neighbor_idx)):
            idx_j = neighbor_idx[j]
            if idx_j in ignored_neighbor:
                Eta = np.array([0, 0, 0, 0, 0, 0])
            else:
                Eta = np.matmul(MinvHrs, b_data[i][j])
            #i[indexdx_i].append(idx_j)
            EtaDx[idx_i,idx_j] = Eta[1]
            EtaDy[idx_i,idx_j] = Eta[2]
            EtaDxx[idx_i,idx_j] = Eta[3]
            EtaDxy[idx_i,idx_j] = Eta[4]
            EtaDyy[idx_i,idx_j] = Eta[5]
            
    return EtaDx, EtaDy, EtaDxx, EtaDxy, EtaDyy
        