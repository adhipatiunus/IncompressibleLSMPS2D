#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 19:05:47 2022

@author: adhipatiunus
"""
import numpy as np

def neighbor_search_cell_list(node_x, node_y, index, cell_size, y_max, y_min, x_max, x_min):
    nrows = int((y_max - y_min) / cell_size) + 1
    ncols = int((x_max - x_min) / cell_size) + 1
    #print(cell_size)
    #print(nrows)

    cell = [[[] for i in range(ncols)] for j in range(nrows)]

    N = len(node_x)

    for i in range(N):
        listx = int((node_x[i] - x_min) / cell_size)
        listy = int((node_y[i] - y_min) / cell_size)
        #print(particle.x[i])
        cell[listx][listy].append(index[i])
        
    neighbor = [[] for i in range(N)]

    for i in range(nrows):
        for j in range(ncols):
            for pi in cell[i][j]:
                neigh_row, neigh_col = i - 1, j - 1
                push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i - 1, j
                push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i - 1, j + 1
                push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i, j - 1
                push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i, j
                push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i, j + 1
                push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i + 1, j - 1
                push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i + 1, j
                push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i + 1, j + 1
                push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
    
    return neighbor
                            
def push_back_particle(neighbor, node_x, node_y, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols):
    if neigh_row >= 0 and neigh_col >= 0 and neigh_row < nrows and neigh_col < ncols:
        for pj in cell[neigh_row][neigh_col]:
            if distance(node_x[pi], node_x[pj], node_y[pi], node_y[pj]) < cell_size:
                neighbor[pi].append(pj)
                    
def distance(x1, x2, y1, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)