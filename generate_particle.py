#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:45:13 2022

@author: adhipatiunus
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_particles_singleres(xmin, xmax, ymin, ymax, sigma, R):
    h = sigma
    
    lx = xmax - xmin
    ly = ymax - ymin

    nx = int(lx / h) + 1
    ny = int(ly / h) + 1
    
    # 1. East
    y_east = np.linspace(ymin, ymax, ny)
    x_east = np.ones_like(y_east) * xmax
    normal_x_east = np.ones_like(y_east) * 1.0
    normal_y_east = np.zeros_like(y_east)
    
    n_east = len(x_east)
    
    # 2. West
    y_west = np.linspace(ymin, ymax, ny)
    x_west = np.ones_like(y_west) * xmin
    normal_x_west = np.ones_like(y_west) * -1.0
    normal_y_west = np.zeros_like(y_west)
    
    n_west = n_east + len(x_west)

    # 3. North
    x_north = np.linspace(xmin+h, xmax-h, nx-2)
    y_north = np.ones_like(x_north) * ymax
    normal_x_north = np.zeros_like(x_north)
    normal_y_north = np.ones_like(x_north) * 1.0
    
    n_north = n_west + len(x_north)
    
    # 4. South
    x_south = np.linspace(xmin+h, xmax-h, nx-2)
    y_south = np.ones_like(x_south) * ymin
    normal_x_south = np.zeros_like(x_south)
    normal_y_south = np.ones_like(x_south) * -1.0

    normal_x_bound = np.concatenate((normal_x_east, normal_x_west, normal_x_north, normal_x_south))
    normal_y_bound = np.concatenate((normal_y_east, normal_y_west, normal_y_north, normal_y_south))

    node_x = np.concatenate((x_east, x_west, x_north, x_south))
    node_y = np.concatenate((y_east, y_west, y_north, y_south))

    n_south = n_north + len(x_south)
    
    n_boundary = np.array([n_east, n_west, n_north, n_south])
    
    # Inner particles
    lx = (xmax - h) - (xmin + h)
    ly = (ymax - h) - (ymin + h)

    nx = int(lx / h) + 1
    ny = int(ly / h) + 1
    
    x = np.linspace(xmin + h, xmax - h, nx)
    y = np.linspace(ymin + h, ymax - h, ny)
    
    x, y = np.meshgrid(x, y)
    
    x = x.flatten()
    y = y.flatten()
    
    node_x = np.concatenate((node_x, x))
    node_y = np.concatenate((node_y, y))
    
    diameter = h * np.ones_like(node_x)
    
    N = len(node_x)
    
    index = np.arange(N)
    
    return node_x, node_y, normal_x_bound, normal_y_bound, n_boundary, index, diameter

def generate_particles(xmin, xmax, x_center, ymin, ymax, y_center, sigma, R):
    h1 = 1
    h2 = 1/2
    h3 = 1/4
    h4 = 1/8
    h5 = 0.05
    h6 = 0.025
    h7 = 0.0125

    h = h5 * sigma

    lx = xmax - xmin
    ly = ymax - ymin

    nx = int(lx / h) + 1
    ny = int(ly / h) + 1

    # Generate Boundary Particle

    # 1. East
    y_east = np.linspace(ymin, ymax, ny)
    x_east = np.ones_like(y_east) * xmax
    normal_x_east = np.ones_like(y_east) * -1.0
    normal_y_east = np.zeros_like(y_east)
    n_east = len(x_east)
    
    # 2. West
    y_west = np.linspace(ymin, ymax, ny)
    x_west = np.ones_like(y_west) * xmin
    normal_x_west = np.ones_like(y_west) * 1.0
    normal_y_west = np.zeros_like(y_west)
    n_west = n_east + len(x_west)

    # 3. North
    x_north = np.linspace(xmin, xmax, nx)
    y_north = np.ones_like(x_north) * ymax
    normal_x_north = np.ones_like(x_north) * -1.0
    normal_y_north = np.zeros_like(x_north)
    n_north = n_west + len(x_north)
    
    # 4. South
    x_south = np.linspace(xmin, xmax, nx)
    y_south = np.ones_like(x_south) * ymin
    normal_x_south = np.ones_like(x_south) * 1.0
    normal_y_south = np.zeros_like(x_south)

    normal_x_bound = np.concatenate((normal_x_east, normal_x_west, normal_x_north, normal_x_south))
    normal_y_bound = np.concatenate((normal_y_east, normal_y_west, normal_y_north, normal_y_south))

    node_x = np.concatenate((x_east, x_west, x_north, x_south))
    node_y = np.concatenate((y_east, y_west, y_north, y_south))
    n_south = n_north + len(x_south)

    n_bound = len(node_x)
    diameter = h * np.ones(n_bound)
    
    # Inner Sphere
    # First layer
    h = h5 * sigma
    R_in = 0
    R_out = R - 4 * h
    sphere_x, sphere_y, sp = generate_node_spherical(x_center, y_center, R_in, R_out, h)

    node_x = np.concatenate((node_x, sphere_x))
    node_y = np.concatenate((node_y, sphere_y))
    diameter = np.concatenate((diameter, sp))
    
    # Second layer
    h = h6 * sigma
    R_in = R_out
    R_out = R - 1 * h
    sphere_x, sphere_y, sp = generate_node_spherical(x_center, y_center, R_in, R_out, h)

    node_x = np.concatenate((node_x, sphere_x))
    node_y = np.concatenate((node_y, sphere_y))
    diameter = np.concatenate((diameter, sp))
    
    # Third layer    
    h = h7 * sigma
    R_in = R_out
    R_out = R
    sphere_x, sphere_y, sp = generate_node_spherical(x_center, y_center, R_in, R_out, h)

    node_x = np.concatenate((node_x, sphere_x))
    node_y = np.concatenate((node_y, sphere_y))
    diameter = np.concatenate((diameter, sp))
    
    # Outside sphere
    # First layer
    h = h7 * sigma
    n_layer = 3
    R_in = R_out
    R_out = R + n_layer * h
    sphere_x, sphere_y, sp = generate_node_spherical(x_center, y_center, R_in, R_out, h)

    node_x = np.concatenate((node_x, sphere_x))
    node_y = np.concatenate((node_y, sphere_y))
    diameter = np.concatenate((diameter, sp))
    
    # Second layer
    h = h6 * sigma
    n_layer = 10
    R_in = R_out
    R_out = R + n_layer * h
    sphere_x, sphere_y, sp = generate_node_spherical(x_center, y_center, R_in, R_out, h)

    node_x = np.concatenate((node_x, sphere_x))
    node_y = np.concatenate((node_y, sphere_y))
    diameter = np.concatenate((diameter, sp))
    
    # Intermediate
    # Inner boundary: circle
    # Outer boundary: rectangle
    h = h6 * sigma
    n_layer = 10
    X_MIN = xmin + h
    X_MAX = xmax - h
    Y_MIN = ymin + h
    Y_MAX = ymax - h

    nx = int((X_MAX - X_MIN) / h) + 1
    ny = int((Y_MAX - Y_MIN) / h) + 1

    x = np.linspace(X_MIN, X_MAX, nx)
    y = np.linspace(Y_MIN, Y_MAX, ny)

    x_3d, y_3d = np.meshgrid(x, y)

    rec_x = x_3d.flatten()
    rec_y = y_3d.flatten()

    delete_inner = (rec_x - x_center)**2 + (rec_y - y_center)**2 <= R_out

    X_MIN = (x_center - R_out) - n_layer * h
    X_MAX = (x_center + R_out) + 5 * n_layer * h
    Y_MIN = (y_center - R_out) - n_layer * h
    Y_MAX = (y_center + R_out) + n_layer * h

    delete_outer = (rec_x < X_MIN) + (rec_x > X_MAX) + (rec_y < Y_MIN) + (rec_y > Y_MAX)
    delete_node = delete_inner + delete_outer

    rec_x = rec_x[~delete_node]
    rec_y = rec_y[~delete_node]
    sp = h * np.ones_like(rec_x)

    node_x = np.concatenate((node_x, rec_x))
    node_y = np.concatenate((node_y, rec_y))
    diameter = np.concatenate((diameter, sp))
    
    # Box of nodes
    # First box
    h = h5 * sigma
    n_layer = 3
    x_bound_min = X_MIN
    x_bound_max = X_MAX
    y_bound_min = Y_MIN
    y_bound_max = Y_MAX

    rec_x, rec_y, sp = generate_node_box(xmin, xmax, ymin, ymax, x_bound_min, x_bound_max, y_bound_min, y_bound_max, n_layer, h)

    node_x = np.concatenate((node_x, rec_x))
    node_y = np.concatenate((node_y, rec_y))
    diameter = np.concatenate((diameter, sp))

    x_bound_min = x_bound_min - n_layer * h
    x_bound_max = x_bound_max + n_layer * h
    y_bound_min = y_bound_min - n_layer * h
    y_bound_max = y_bound_max + n_layer * h
    
    # Second box
    h = h5 * sigma
    n_layer = 3
    rec_x, rec_y, sp = generate_node_box(xmin, xmax, ymin, ymax, x_bound_min, x_bound_max, y_bound_min, y_bound_max, n_layer, h)

    node_x = np.concatenate((node_x, rec_x))
    node_y = np.concatenate((node_y, rec_y))
    diameter = np.concatenate((diameter, sp))

    x_bound_min = x_bound_min - n_layer * h
    x_bound_max = x_bound_max + n_layer * h
    y_bound_min = y_bound_min - n_layer * h
    y_bound_max = y_bound_max + n_layer * h
    
    # Third box
    h = h5 * sigma

    nx = int(((xmax - h) - (xmin + h)) / h) + 1
    ny = int(((ymax - h) - (ymin + h)) / h) + 1
    
    x = np.linspace(xmin + h, xmax - h, nx)
    y = np.linspace(ymin + h, ymax - h, ny)

    x_3d, y_3d = np.meshgrid(x, y)

    rec_x = x_3d.flatten()
    rec_y = y_3d.flatten()

    delete_inner = (rec_x >= x_bound_min) * (rec_x <= x_bound_max) \
                    * (rec_y >= y_bound_min) * (rec_y <= y_bound_max)

    rec_x = rec_x[~delete_inner]
    rec_y = rec_y[~delete_inner]
    sp = h * np.ones_like(rec_x)

    node_x = np.concatenate((node_x, rec_x))
    node_y = np.concatenate((node_y, rec_y))
    diameter = np.concatenate((diameter, sp))

    N = len(node_x)
    index = np.arange(0, N)

    boundary = np.full(N, False)
    boundary[:n_bound] = True
    node_z = np.zeros_like(node_x)
    
    n_boundary = np.array([n_east, n_west, n_north, n_south])
    
    return node_x, node_y, node_z, normal_x_bound, normal_y_bound, n_boundary, index, diameter

def generate_node_spherical(x_center, y_center, R_in, R_out, h):
    x_min = x_center - 2 * R_out
    x_max = x_center + 2 * R_out
    y_min = y_center - 2 * R_out
    y_max = y_center + 2 * R_out
    
    lx = x_max - x_min
    ly = y_max - y_min

    nx = int(lx / h) + 1
    ny = int(ly / h) + 1
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    
    x_2d, y_2d= np.meshgrid(x, y)
    
    node_x = x_2d.flatten()
    node_y = y_2d.flatten()
    
    delete_inner = (node_x - x_center)**2 + (node_y - y_center)**2 <= R_in
    delete_outer = (node_x - x_center)**2 + (node_y - y_center)**2 > R_out
    delete_node = delete_inner + delete_outer
    
    node_x = node_x[~delete_node]
    node_y = node_y[~delete_node]
    sp = h * np.ones_like(node_x)
    
    return node_x, node_y, sp

def generate_node_box(x_min, x_max, y_min, y_max, x_bound_min, x_bound_max, y_bound_min, y_bound_max, n_layer, h):    
    nx = int((x_max - x_min) / h) + 1
    ny = int((y_max - y_min) / h) + 1
    
    safety = 1e-10
    
    x_outer_min = x_bound_min - n_layer * h
    x_outer_max = x_bound_max + n_layer * h
    y_outer_min = y_bound_min - n_layer * h
    y_outer_max = y_bound_max + n_layer * h
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    
    x_3d, y_3d = np.meshgrid(x, y)
    
    node_x = x_3d.flatten()
    node_y = y_3d.flatten()
    
    delete_inner = (node_x >= x_bound_min) * (node_x <= x_bound_max) \
                    *(node_y >= y_bound_min) * (node_y <= y_bound_max) 
                    
    delete_outer = (node_x < x_outer_min - safety) + (node_x > x_outer_max + safety) \
                    + (node_y < y_outer_min - safety) + (node_y > y_outer_max + safety)
                    
    delete = delete_inner + delete_outer
            
    node_x = node_x[~delete]
    node_y = node_y[~delete]
    sp = h * np.ones_like(node_x)
    
    return node_x, node_y, sp

