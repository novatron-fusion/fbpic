# Copyright 2020, FBPIC contributors
# Authors: Kristoffer Lindvall
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the Perfectly Magnetic Conductor (PMC) class, which set the 
magnetic fields to 0 inside the wall
"""
from scipy.constants import c
import math as m
import numpy as np
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_2d, compile_cupy

#@cuda.jit( device=True, inline=True )
def P00( x, h, L ):
    return (-(x - h)*(x + L)**2 / ( h * L**2 ) )


#@cuda.jit( device=True, inline=True )
def P01( x, h, L ):
    return ( x*(x + L)**2 / ( h * (h + L)**2 ) )

class PMC(object):

    def __init__( self, z_start=None, h=None, L=None, gamma_boost=None, m='all', eta=0.25):
        """
        Initialize a perfect magnetic conductor.

        The wall reflects the magnetic field

        Parameters
        ----------
        rwall : float
            radial position of the wall

        gamma_boost : float
            For boosted-frame simulation: Lorentz factor of the boost

        m : int or list of ints
            Specify the field modes to set to zero
            By default, takes all modes to zero
        """

        # Calculate array of normal vectors
        self.z_start = z_start
        self.z_end = z_start - L
        self.L = L
        self.h = h
        self.eta = eta
        self.gamma_boost = gamma_boost
        if m == 'all':
            self.modes = None
        elif isinstance(m, int):
            self.modes = [m]
        elif isinstance(m, list):
            self.modes = m
        else:
            raise TypeError('m should be an int or a list of ints.')

    def save_old_Bfield( self, interp ):
        for i, grid in enumerate(interp):

            if self.modes is not None:
                if i not in self.modes:
                    continue

            self.Br_o = getattr( grid, 'Br')
            self.Bt_o = getattr( grid, 'Bt')
            self.Bz_o = getattr( grid, 'Bz')



    def reflect_Bfield_at_pmc(  self, interp, comm, t_boost ):
        """
        Reflect B field at perfect magnetic conductor
        """
        # Calculate indices in z between which the field should be set to 0
        zmin, zmax = comm.get_zmin_zmax( local=False,
                        with_guard=True, with_damp=True, rank=comm.rank)

        z = cupy.linspace(zmin, zmax, interp[0].Nz)
        r = cupy.linspace(0., interp[0].rmax, interp[0].Nr)
        
        
        im = int( (self.z_start - zmin) / interp[0].dz)
        imax = int( (self.z_start+self.h - zmin) / interp[0].dz)
        imin = int( (self.z_start-self.L - zmin) / interp[0].dz)
        n_cells = 3
        #imin = max( imax - n_cells, 0)

        R, Z = cupy.meshgrid(r,z)

        for i, grid in enumerate(interp):

            if self.modes is not None:
                if i not in self.modes:
                    continue

            Br = getattr( grid, 'Br')
            Bt = getattr( grid, 'Bt')
            Bz = getattr( grid, 'Bz')
            #Jr = getattr( grid, 'Jr')
            #Jt = getattr( grid, 'Jt')
            #Jz = getattr( grid, 'Jz')

            Br[ imin:im+3, :] -= self.eta*(self.Br_o[ imin:im+3, :] \
                    - self.Br_o[ imax, :]*(P01( Z[ imin:im+3, :], self.h, self.L )))
            
            Bt[ imin:im+3, :] -= self.eta*(self.Bt_o[ imin:im+3, :] \
                    - self.Bt_o[ imax, :]*(P01( Z[ imin:im+3, :], self.h, self.L )))  
            Bz[ imin:im+3, :] -= self.eta*(self.Bz_o[ imin:im+3, :] \
                    - self.Bz_o[ im, :]*(P00( Z[imin:im+3, :], self.h, self.L )) \
                    - self.Bz_o[ imax, :]*(P01( Z[ imin:im+3, :], self.h, self.L )))
            """
            Br[ im-n_cells:im-1, :] = cupy.flip(-Br[im+1:im+n_cells, :], axis=0)  # Uses numpy/cupy syntax
            Bt[ im-n_cells:im-1, :] = cupy.flip(-Bt[im+1:im+n_cells, :], axis=0)  # Uses numpy/cupy syntax
            Bz[ im-n_cells:im-1, :] = cupy.flip(Bz[im+1:im+n_cells, :], axis=0)  # Uses numpy/cupy syntax
            """
            Br[ imin:im+3, -1] = 0.
            Bt[ imin:im+3, -1] = 0.
            Bz[ imin:im+3, -1] = 0.
            Br[ imin:im+3, 0] = 0.
            Bt[ imin:im+3, 0] = 0.
            Br[ im, :] = 0.
            Bt[ im, :] = 0.
            
            #Br[ im-n_cells:im-1, :] = cupy.flip(Br[im+1:im+n_cells, :], axis=0)  # Uses numpy/cupy syntax
            #Bt[ im-n_cells:im-1, :] = cupy.flip(Bt[im+1:im+n_cells, :], axis=0)  # Uses numpy/cupy syntax
            #Bz[ im-n_cells:im-1, :] = cupy.flip(-Bz[im+1:im+n_cells, :], axis=0)  # Uses numpy/cupy syntax
            #Et[ im+1,-1] = 0.
            #Er[ im+1, 0] = 0.
            #Et[ im+1, 0] = 0.
            #Ez[ imax, :] = 0


            
            
