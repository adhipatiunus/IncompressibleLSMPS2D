#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 20:58:39 2022

@author: adhipatiunus
"""
import numpy as np
import matplotlib.pyplot as plt

def visualize(x, y, f, diameter, filename):
    ## Scatter Elliptical Particle
    # unit area ellipse
    rx = 1.0
    ry = 1.0
    size = 0.01
    area = rx * ry * np.pi
    theta = np.arange(0, 2 * np.pi + 0.01, 0.1)
    verts = np.column_stack([rx / area * np.cos(theta), ry / area * np.sin(theta)])

    # ================================ #
    # ======== Retrieve Data ========= #
    # ================================ #

    # Read data file
    X = x
    Y = y
    dfx1A = f

    # ================================ #
    # ======== Creating Plot ========= #
    # ================================ #

    # The plot limit 
    ymax = 1
    ymin = 0
    xmax = 1
    xmin = 0

    # Define a figure plot
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111)
    ax.set_aspect(1)    # Make a 1:1 axis scale

    fig.canvas.draw()
    SS = (2.5 *rx* size * ax.get_window_extent().width  / (xmax-xmin+1.) * 72./fig.dpi) ** 2

    plt.scatter(X, Y, s= SS, c=dfx1A, cmap="jet", marker=verts, linewidth=0)  # For mesh & manual grid use

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    cbar = plt.colorbar()

    # To make a x10 scientific notation
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    # Adjust the label
    plt.xlabel("x", size=18)
    plt.ylabel("y", size=18)

    # The label and colorbar tick size
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    cbar.ax.tick_params(labelsize=10)

    # Adjust the tick spacing to be 1
    plt.yticks(np.linspace(ymin, ymax,int(1+ymax-ymin)))
    plt.xticks(np.linspace(xmin, xmax,int(1+xmax-xmin)))

    # Saving figure
    save_name = filename
    # plt.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.show()