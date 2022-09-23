# Copyright 2020, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Alberto de la Ossa
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the Perfectly Conducting Wall (PCW) class, which set the 
fields to 0 inside the wall
"""
from scipy.constants import c
import math as m
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_2d, compile_cupy


@cuda.jit( device=True, inline=True )
def ray_casting(x, y, polygon):
    n = len(polygon)
    count = 0
    for i in range(n-1):
        x1 = polygon[i][0]
        x2 = polygon[i+1][0]
        y1 = polygon[i][1]
        y2 = polygon[i+1][1]

        if (y < y1) != (y < y2) \
            and x < (x2-x1) * (y-y1) / (y2-y1) + x1:
            count += 1
        
    return(False if count % 2 == 0 else True)


class PCW(object):

    def __init__( self, wall_arr, gamma_boost=None, m='all'):
        """
        Initialize a perfectly conducting wall.

        The wall reflects the fields, by setting the specified
        field modes to 0 in the wall, at each timestep.
        By default, all modes are zeroed.

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
        
        self.wall_arr = cupy.asarray(wall_arr)
        self.gamma_boost = gamma_boost

        if m == 'all':
            self.modes = None
        elif isinstance(m, int):
            self.modes = [m]
        elif isinstance(m, list):
            self.modes = m
        else:
            raise TypeError('m should be an int or a list of ints.')


    def set_fields_to_zero( self, interp ):
        """
        Set the fields to 0 inside/outside wall

        Parameters:
        -----------
        interp: a list of InterpolationGrid objects
            Contains the values of the fields in interpolation space
        """

        # Set fields (E, B) to 0 on CPU or GPU
        for i, grid in enumerate(interp):
            z_grid = grid.zmin + (0.5+cupy.arange(grid.Nz))*grid.dz
            r_grid = grid.rmin + (0.5+cupy.arange(grid.Nr))*grid.dr

            if self.modes is not None:
                if i not in self.modes:
                    continue
            
            fieldlist = ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz']
            if grid.use_pml:
                fieldlist = fieldlist + ['Er_pml', 'Et_pml', 'Br_pml', 'Bt_pml']
            for field in fieldlist:
                arr = getattr( grid, field )
                dim_grid, dim_block = cuda_tpb_bpg_2d( 
                    arr.shape[0], arr.shape[1] )
                # Call kernel
                self.pcw_boundary[ dim_grid, dim_block ](
                     arr, z_grid, r_grid, self.wall_arr)


    @compile_cupy
    def pcw_boundary( F, z, r, wall_arr):
        i, j = cuda.grid(2)
        if i < F.shape[0] and j < F.shape[1]:
            in_poly = ray_casting(z[i], r[j], wall_arr)
            if not in_poly:
                F[i, j] = 0.


    
