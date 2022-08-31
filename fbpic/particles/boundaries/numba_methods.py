# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kristoffer Lindvall
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle boundary methods on the CPU with numba.
"""
import numba
import math as m
from fbpic.utils.threading import njit_parallel, prange

@njit_parallel
def reflect_particles_radially_numba(rmax, x, y, ux, uy, Ntot):
    """
    Reflect particles at rmax radius
    Parameters
    ----------
    rmax : radiall boundary

    x : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    
    y : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)

    ux : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    uy : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    """

    # Particle reflect (in parallel if threading is installed)
    for ip in prange(Ntot):
        r = m.sqrt(x[ip]**2 + y[ip]**2)
        if r > rmax:
            if x[ip] == 0:
                if y[ip] > 0:
                    theta = m.pi / 2
                elif y[ip] == 0:
                    theta = 0.
                elif y[ip] < 0:
                    theta = 3 * m.pi / 2
            else:
                theta = m.atan( y[ip] / x[ip])
            temp = ux[ip]
            ux[ip] = uy[ip]
            uy[ip] = temp
            x[ip] = ( rmax  - (r - rmax ) ) * m.cos(theta)
            y[ip] = ( rmax  - (r - rmax ) ) * m.sin(theta)
    return x, y, ux, uy