# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kristoffer Lindvall
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle boundary methods on the GPU using CUDA.
"""
from numba import cuda
import math as m
from fbpic.utils.cuda import compile_cupy

@compile_cupy
def reflect_particles_radially_cuda(rmax, x, y, ux, uy):
    """
    Reflect particles radially
    Parameters
    ----------
    zmin : left boundary position
    z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    """
    i = cuda.grid(1)
    if i < x.shape[0]:
        r = m.sqrt(x[i]**2 + y[i]**2)
        if r > rmax:
            if x[i] == 0:
                if y[i] > 0:
                    theta = m.pi / 2
                elif y[i] == 0:
                    theta = 0.
                elif y[i] < 0:
                    theta = 3 * m.pi / 2
            else:
                theta = m.atan( y[i] / x[i])
            temp = ux[i]
            ux[i] = uy[i]
            uy[i] = temp
            x[i] = ( rmax  - (r - rmax ) ) * m.cos(theta)
            y[i] = ( rmax  - (r - rmax ) ) * m.sin(theta)

@compile_cupy
def periodic_particles_radially_cuda(rmax, x, y):
    """
    Periodic particles at radial boundary
    Parameters
    ----------
    zmin : left boundary position
    z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    """
    i = cuda.grid(1)
    if i < x.shape[0]:
        r = m.sqrt(x[i]**2 + y[i]**2)
        if r > rmax:
            if x[i] == 0:
                if y[i] > 0:
                    theta = m.pi / 2
                elif y[i] == 0:
                    theta = 0.
                elif y[i] < 0:
                    theta = 3 * m.pi / 2
            else:
                theta = m.atan( y[i] / x[i])
            x[i] = ( rmax  - (r - rmax ) ) * m.cos(theta + m.pi)
            y[i] = ( rmax  - (r - rmax ) ) * m.sin(theta + m.pi)