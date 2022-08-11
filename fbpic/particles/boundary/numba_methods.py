# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle boundary methods on the CPU with numba.
"""
import numba
from fbpic.utils.threading import njit_parallel, prange
# Compile the inline functions for CPU

@njit_parallel
def reflect_particles_left_numba( zmin, z, uz, Ntot ):
    """
    Reflect particles at left boundary

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    """

    # Particle reflect (in parallel if threading is installed)
    for ip in prange(Ntot):
        if z[ip] < zmin:
            uz[ip] *= -1
            z[ip] = ( zmin - z[ip] ) + zmin

    return z, uz

@njit_parallel
def reflect_particles_right_numba( zmax, z, uz, Ntot ):
    """
    Reflect particles at left boundary

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    """

    # Particle reflect (in parallel if threading is installed)
    for ip in prange(Ntot):
        if z[ip] > zmax:
            uz[ip] *= -1
            z[ip] = zmax - ( z[ip] - zmax )

    return z, uz

@njit_parallel
def bounce_particles_left_numba( zmin, x, y, z, ux, uy, uz, Ntot ):
    """
    Bounce particles at left boundary

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    """
    for ip in prange(Ntot):
        if z[ip] < zmin:
            ux[ip] *= -1
            uy[ip] *= -1
            uz[ip] *= -1
            x[ip] *= -1
            y[ip] *= -1
            z[ip] = ( zmin - z[ip] ) + zmin

    return x, y, z, ux, uy, uz

@njit_parallel
def bounce_particles_right_numba( zmax, x, y, z, ux, uy, uz, Ntot ):
    """
    Bounce particles at right boundary

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    """
    for ip in prange(Ntot):
        if z[ip] > zmax:
            ux[ip] *= -1
            uy[ip] *= -1
            uz[ip] *= -1
            x[ip] *= -1
            y[ip] *= -1
            z[ip] = zmax - ( z[ip] - zmax )
    
    return x, y, z, ux, uy, uz
