#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:10:59 2022

@author: adhipatiunus
"""

import numpy as np
import json 
# from matplotlib import pyplot as plt

# %% 
def get_cell_neigh(idxi, idxj, idxk, n_cell_x, n_cell_y, n_cell_z, N_CELL_APART=1):
    """
    Generates cell neighbors of a cell given its location(idxi, idxj, idxk)

    Args:
        idxi ([int]): cell location of the particle p in the x-axis
        idxj ([int]): cell location of the particle p in the y-axis
        idxk ([int]): cell location of the particle p in the z-axis
        n_cell_x ([int]): number of cells in the x-dir
        n_cell_y ([int]): number of cells in the y-dir
        n_cell_z ([int]): number of cells in the z-dir
        N_CELL_APART ([int]): number of cell apart

    Returns:
        cell_neighbors_2d ([2d np array]) : [idxi idxj] index of the neighbors of the corresponding cell
        n_cell_neigh ([int]) : number of neighbors of the corresponding cell
    """
    max_cell_neigh = (2*N_CELL_APART+1)**3
    cell_neighbors_2d = np.empty((max_cell_neigh, 3), dtype=int)
    n_cell_neigh = int(0)
    for i in range(-N_CELL_APART, N_CELL_APART+1):
        if (idxi+i)>=0 and (idxi+i)<n_cell_x: # if the neigh idx is [0,max]
            for j in range(-N_CELL_APART, N_CELL_APART+1):
                if (idxj+j)>=0 and (idxj+j)<n_cell_y: # if the neigh idx is [0,max]
                    for k in range(-N_CELL_APART, N_CELL_APART+1):
                        if (idxk+k)>=0 and (idxk+k)<n_cell_z: # if the neigh idx is [0,max]
                            cell_neighbors_2d[n_cell_neigh,0] = idxi+i
                            cell_neighbors_2d[n_cell_neigh,1] = idxj+j
                            cell_neighbors_2d[n_cell_neigh,2] = idxk+k
                            n_cell_neigh = n_cell_neigh + 1


    return cell_neighbors_2d, n_cell_neigh

def get_cell_neigh_dict(n_cell_x, n_cell_y, n_cell_z, N_CELL_APART=1):
    """Generates dictionary of cell neighbors

    Args:
        n_cell_x (int): number of 
        n_cell_y ([type]): [description]
        n_cell_z ([type]): [description]
        N_CELL_APART (int, optional): number of cell apart considered as neighbors. Defaults to 1.

    Returns:
        cell_neigh_dict_5d(5d np.array): dictionary of cell neighbors
        n_cell_neigh_dict_3d(3d np.array): dictionary of n_neigh
    """
    max_cell_neigh = (2*N_CELL_APART+1)**3
    cell_neigh_dict_5d = np.empty((n_cell_x, n_cell_y, n_cell_z,max_cell_neigh,3), dtype=int)
    n_cell_neigh_dict_3d = np.zeros((n_cell_x, n_cell_y, n_cell_z), dtype=int)
    for i in range(n_cell_x):
        for j in range(n_cell_y):
            for k in range(n_cell_z):
                cell_neighbors_2d, n_cell_neigh = get_cell_neigh(i, j, k, n_cell_x, n_cell_y, n_cell_z, N_CELL_APART)
                cell_neigh_dict_5d[i,j,k] = np.copy(cell_neighbors_2d)
                n_cell_neigh_dict_3d[i,j,k] = n_cell_neigh
    
    return cell_neigh_dict_5d, n_cell_neigh_dict_3d

# %%
def direct(nodes_2d, n_bound, input_rc_, max_n_neigh):
    """Generates the neighbors of every particle in the domain using the direct search method.
    # ! The coordinates must be in the form of [x|y|z] with boundary particles sorted in the top of the row
    Args:
        nodes_2d (2D np array) : Nodes coordinates [x y]
        n_bound (int): number of boundary particles
        input_rc (float): cutoff radius for inner particles
        input_rc_bound (float): cutoff radius for boundary particles
        max_n_neigh (float) : approx. max number of neighbors
    """
    skin = 1e-10 # safety factor

    n_total = nodes_2d.shape[0]
    # initialize matrix to store neighbors idx and number of neighs
    # neighbors_2d = np.empty((n_total,max_n_neigh), dtype=int) 
    neighbors_2d = [[] for i in range(n_total)]
    n_neigh_ = np.zeros((n_total) , dtype=int)

    # performing neighbor search
    for i in range(n_total):
        rc_i = input_rc_[i] 

        for j in range(n_total):
            if i >= n_bound:
                rc = min(rc_i, input_rc_[j]) # to ensure symmetric particle interaction for inner nodes
            else:
                rc = rc_i 
            if np.linalg.norm(nodes_2d[i,:]-nodes_2d[j,:]) <= rc+skin:
                # neighbors_2d[i,n_neigh_[i]] = j
                neighbors_2d[i].append(j)
                n_neigh_[i] += 1                       
    
    # neighbors_2d = neighbors_2d[:, 0:max(n_neigh_)]
    return neighbors_2d, n_neigh_        

# %%
def verlet(nodes_2d, n_bound, input_rc_, max_n_neigh):
    """
    Search the neighbors of every particle in the domain using the verlet list procedure.
    # ! The coordinates must be in the form of [x|y] with boundary particles sorted in the top of the row
    Args:
        nodes_2d (2D np array) : Nodes coordinates [x y]
        input_rc (float): cutoff radius for inner particles
        input_rc_bound (float): cutoff radius for boundary particles
        max_n_neigh (float) : approx. max number of neighbors

    Returns:
        neighbors_2d (2d np array) : index of the neighbors of each particle
        n_neigh_ (1d np array) : number of neighbors for each particle
    """
    skin = 1e-10 # multiplier to accomodate jiggling particles

    n_total = nodes_2d.shape[0]
    # initialize matrix to store neighbors idx and number of neighs
    # neighbors_2d = np.empty((n_total,max_n_neigh), dtype=int) 
    neighbors_2d = [[] for i in range(n_total)]
    n_neigh_ = np.zeros((n_total) , dtype=int)

    #* 2 kinds of problem may occur
    # 1st - when lx % rc == 0 (meaning the cells perfectly fit the domain),
    # in this case the rightmost particle may be placed outside the rightmost cell.
    # solution: add a safety factor to the domain length, so that the cells covers a bit more of the domain. 
    # 2nd - when a particle is located on the edge of a cell which has a length exactly equal rc.
    # if its neighbor is at a length rc to that particle, that neighbor may not be captured.
    # solution to this: give a safety factor to the cell's size 

    # initialize cell list
    maxx, maxy, maxz = max(nodes_2d[:,0]), max(nodes_2d[:,1]), max(nodes_2d[:,2])
    minx, miny, minz = min(nodes_2d[:,0]), min(nodes_2d[:,1]), min(nodes_2d[:,2])
    lx = maxx - minx + 1e-10 # domain length
    ly = maxy - miny + 1e-10 # domain length
    lz = maxz - minz + 1e-10 # domain length

    max_rc = max(input_rc_) # cell size
    cell_size = max_rc*1.01 +skin
    n_cell_x = int(np.ceil(lx/cell_size)) # number of cells
    n_cell_y = int(np.ceil(ly/cell_size))
    n_cell_z = int(np.ceil(lz/cell_size))

    # cell_4d : 4d array for storing what particles are inside a cell
    # n_inside_cell_3d : number of particles inside a cell
    #! multiplier due to variable resolution = (h_max/h_min)**2
    mult = 4**2 # =(h_max/h_min)**2
    cell_4d = np.empty((n_cell_x,n_cell_y,n_cell_z,max_n_neigh*mult), dtype=int)
    n_inside_cell_3d = np.zeros((n_cell_x,n_cell_y,n_cell_z), dtype=int)
    # initialize cell_loc_2d: cell location [idxi, idxj, idxk] for every particle 
    cell_loc_2d = np.empty_like(nodes_2d, dtype=int)

    # perform cell list : distributing particles into cells
    for i in range(n_total):
        idxi = int(np.floor(nodes_2d[i,0]/cell_size)) 
        idxj = int(np.floor(nodes_2d[i,1]/cell_size))
        idxk = int(np.floor(nodes_2d[i,2]/cell_size))
        cell_loc_2d[i,0] = idxi
        cell_loc_2d[i,1] = idxj
        cell_loc_2d[i,2] = idxk
        cell_4d[idxi,idxj,idxk,n_inside_cell_3d[idxi,idxj,idxk]] = i
        n_inside_cell_3d[idxi,idxj,idxk] += 1

    # cell neighbor search
    cell_neigh_dict_5d, n_cell_neigh_dict_3d = get_cell_neigh_dict(n_cell_x, n_cell_y, n_cell_z, N_CELL_APART=1)
    
    # particle neighbor search
    for i in range(n_total):
        # N_CELL_APART = 1 # not-really-an-input because it must be 1 actually
        # cutoff radius
        rc_i = input_rc_[i]        

        # cell location of particle i
        idxi = cell_loc_2d[i,0]
        idxj = cell_loc_2d[i,1]
        idxk = cell_loc_2d[i,2]

        # # cell-neighbor search
        # cell_neighbors_2d, n_cell_neigh = get_cell_neigh(idxi, idxj, idxk, n_cell_x, n_cell_y, n_cell_z, N_CELL_APART)

        # particle-neighbor search
        for l in range(n_cell_neigh_dict_3d[idxi,idxj,idxk]):
            neigh_idxi = cell_neigh_dict_5d[idxi,idxj,idxk,l,0]
            neigh_idxj = cell_neigh_dict_5d[idxi,idxj,idxk,l,1]
            neigh_idxk = cell_neigh_dict_5d[idxi,idxj,idxk,l,2]
            for j_th in range(n_inside_cell_3d[neigh_idxi, neigh_idxj, neigh_idxk]):
                # obtain index of the j-th particle inside the l-th neighboring cell
                j = cell_4d[neigh_idxi, neigh_idxj, neigh_idxk, j_th]
                if i >= n_bound:
                    rc = min(rc_i, input_rc_[j]) # to ensure symmetric particle interaction for inner nodes
                else:
                    rc = rc_i 
                if np.linalg.norm(nodes_2d[i,:]-nodes_2d[j,:]) <= rc+skin:
                    # neighbors_2d[i,n_neigh_[i]] = j
                    neighbors_2d[i].append(j)
                    n_neigh_[i] = n_neigh_[i] + 1
         
    # neighbors_2d = neighbors_2d[:, 0:max(n_neigh_)]
    return neighbors_2d, n_neigh_


# %%
def multiple_verlet(nodes_2d, n_bound, input_rc_):
    """
    Search the neighbors of every particle in the domain using the verlet list procedure.
    # ! The coordinates must be in the form of [x|y] with boundary particles sorted in the top of the row
    Args:
        nodes_2d (2D np array) : Nodes coordinates [x y]
        input_rc (float): cutoff radius for inner particles
        input_rc_bound (float): cutoff radius for boundary particles
        upwind (bool) : whether or not to find upwinding neighbors

    Returns:
        neighbors_2d (2d np array) : index of the neighbors of each particle
        n_neigh_ (1d np array) : number of neighbors for each particle
    """
    skin = 1e-10 # multiplier to accomodate jiggling particles

    n_total = nodes_2d.shape[0]
    # initialize matrix to store neighbors idx and number of neighs
    # neighbors_2d = np.empty((n_total,max_n_neigh), dtype=int) 
    neighbor_all = [[] for i in range(n_total)]
    n_neigh_ = np.zeros((n_total) , dtype=int)

    #* 2 kinds of problem may occur
    # 1st - when lx % rc == 0 (meaning the cells perfectly fit the domain),
    # in this case the rightmost particle may be placed outside the rightmost cell.
    # solution: add a safety factor to the domain length, so that the cells covers a bit more of the domain. 
    # 2nd - when a particle is located on the edge of a cell which has a length exactly equal rc.
    # if its neighbor is at a length rc to that particle, that neighbor may not be captured.
    # solution to this: give a safety factor to the cell's size 

    # initialize cell list
    maxx, maxy, maxz = max(nodes_2d[:,0]), max(nodes_2d[:,1]), max(nodes_2d[:,2])
    minx, miny, minz = min(nodes_2d[:,0]), min(nodes_2d[:,1]), min(nodes_2d[:,2])
    lx = maxx - minx + 1e-10 # domain length
    ly = maxy - miny + 1e-10 # domain length
    lz = maxz - minz + 1e-10 # domain length

    # first cutoff radius

    rc_prev = input_rc_[0] # cell size
    cell_size = rc_prev*1.01 +skin
    n_cell_x = int(np.ceil(lx/cell_size)) # number of cells
    n_cell_y = int(np.ceil(ly/cell_size))
    n_cell_z = int(np.ceil(lz/cell_size))

    # cell_4d : 4d array for storing what particles are inside a cell
    # n_inside_cell_3d : number of particles inside a cell
    cell_4d = [[[[]for k in range(n_cell_z) ]for j in range(n_cell_y) ] for i in range(n_cell_x) ]
    n_inside_cell_3d = np.zeros((n_cell_x,n_cell_y,n_cell_z), dtype=int)
    # initialize cell_loc_2d: cell location [idxi, idxj, idxk] for every particle 
    cell_loc_2d = np.empty_like(nodes_2d, dtype=int)

    # perform cell list : distributing particles into cells
    for i in range(n_total):
        idxi = int(np.floor(nodes_2d[i,0]/cell_size)) 
        idxj = int(np.floor(nodes_2d[i,1]/cell_size))
        idxk = int(np.floor(nodes_2d[i,2]/cell_size))
        cell_loc_2d[i,0] = idxi
        cell_loc_2d[i,1] = idxj
        cell_loc_2d[i,2] = idxk
        # cell_4d[idxi,idxj,idxk,n_inside_cell_3d[idxi,idxj,idxk]] = i
        cell_4d[idxi][idxj][idxk].append(i)
        n_inside_cell_3d[idxi,idxj,idxk] += 1

    # cell neighbor search
    cell_neigh_dict_5d, n_cell_neigh_dict_3d = get_cell_neigh_dict(n_cell_x, n_cell_y, n_cell_z, N_CELL_APART=1)
    
    # # average spacing
    # h_l1_ = np.zeros(n_total)
    # h_l2_ = np.zeros(n_total)

    # particle neighbor search
    for i in range(n_total):
        print('Particle ' + str(i) + ' / ' + str(n_total))
        # cutoff radius
        rc_i = input_rc_[i]
        
        if rc_i != rc_prev:            
            cell_size = rc_i*1.01 +skin
            n_cell_x = int(np.ceil(lx/cell_size)) # number of cells
            n_cell_y = int(np.ceil(ly/cell_size))
            n_cell_z = int(np.ceil(lz/cell_size))

            # cell_4d : 4d array for storing what particles are inside a cell
            # n_inside_cell_3d : number of particles inside a cell
            # cell_4d = np.empty((n_cell_x,n_cell_y,n_cell_z,max_n_inside_cell), dtype=int)
            cell_4d = [[[[]for k in range(n_cell_z) ]for j in range(n_cell_y) ] for i in range(n_cell_x) ]
            n_inside_cell_3d = np.zeros((n_cell_x,n_cell_y,n_cell_z), dtype=int)
            # initialize cell_loc_2d: cell location [idxi, idxj, idxk] for every particle 
            cell_loc_2d = np.empty_like(nodes_2d, dtype=int)

            # perform cell list : distributing particles into cells
            for m in range(n_total):
                idxi = int(np.floor(nodes_2d[m,0]/cell_size)) 
                idxj = int(np.floor(nodes_2d[m,1]/cell_size))
                idxk = int(np.floor(nodes_2d[m,2]/cell_size))
                cell_loc_2d[m,0] = idxi
                cell_loc_2d[m,1] = idxj
                cell_loc_2d[m,2] = idxk
                # cell_4d[idxi,idxj,idxk,n_inside_cell_3d[idxi,idxj,idxk]] = m
                cell_4d[idxi][idxj][idxk].append(m)
                n_inside_cell_3d[idxi,idxj,idxk] += 1

            # cell neighbor search
            cell_neigh_dict_5d, n_cell_neigh_dict_3d = get_cell_neigh_dict(n_cell_x, n_cell_y, n_cell_z, N_CELL_APART=1)

            rc_prev = rc_i                   

        # cell location of particle i
        idxi = cell_loc_2d[i,0]
        idxj = cell_loc_2d[i,1]
        idxk = cell_loc_2d[i,2]

        # # cell-neighbor search
        # cell_neighbors_2d, n_cell_neigh = get_cell_neigh(idxi, idxj, idxk, n_cell_x, n_cell_y, n_cell_z, N_CELL_APART)

        # particle-neighbor search
        for l in range(n_cell_neigh_dict_3d[idxi,idxj,idxk]):
            neigh_idxi = cell_neigh_dict_5d[idxi,idxj,idxk,l,0]
            neigh_idxj = cell_neigh_dict_5d[idxi,idxj,idxk,l,1]
            neigh_idxk = cell_neigh_dict_5d[idxi,idxj,idxk,l,2]
            for j_th in range(n_inside_cell_3d[neigh_idxi, neigh_idxj, neigh_idxk]):
                # obtain index of the j-th particle inside the l-th neighboring cell
                # j = cell_4d[neigh_idxi, neigh_idxj, neigh_idxk, j_th]
                j = cell_4d[neigh_idxi][neigh_idxj][neigh_idxk][j_th]
                if i >= n_bound:
                    rc = min(rc_i, input_rc_[j]) # to ensure symmetric particle interaction for inner nodes
                else:
                    rc = rc_i 
                ij_ = nodes_2d[i,:] - nodes_2d[j,:] # vector from particle j to particle i
                l2 = np.linalg.norm(ij_)
                if l2 <= rc+skin:
                    # neighbors_2d[i,n_neigh_[i]] = j
                    neighbor_all[i].append(j)
                    n_neigh_[i] = n_neigh_[i] + 1
                    # h_l2_[i] += l2 # compute and add l2 distance of each neighbor 
                    # h_l1_[i] += np.linalg.norm(ij_,1)  # compute and add l1 distance of each neighbor 
    # np.save("h_l1_.npy", h_l1_)
    # np.save("h_l2_.npy", h_l2_)
    # h_ = np.load('h_.npy')
    # fig, ax = plt.subplots(3 )
    # ax[0].plot(h_ )
    # ax[1].plot(h_l1_ )
    # ax[2].plot(h_l2_ )

    # neighbors_2d = neighbors_2d[:, 0:max(n_neigh_)]

    # save one-sided (upwind) neighbors
    with open('neighbor.txt', 'w') as file:
        json.dump(neighbor_all, file)    
    
    return neighbor_all, n_neigh_
    

    # neighbors_xpos_2d, neighbors_xneg_2d, neighbors_ypos_2d, neighbors_yneg_2d, neighbors_zpos_2d, neighbors_zneg_2d, 
    #return neighbors_2d, n_neigh_
