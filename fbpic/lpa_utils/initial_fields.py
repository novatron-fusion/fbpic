# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines methods to directly set initial fields in the Simulation box
"""
import numpy as np
from scipy.constants import c
from scipy import interpolate
from fbpic.fields import Fields

def add_initial_B_field_m_0( sim, Br, Bt, Bz):
    """
    Add an initial magnetic field directly to the mesh

    Note:
    -----------
    This function only sets the m=0 component of the 
    magnetic field

    Parameters:
    -----------
    sim: a Simulation object
        The structure that contains the simulation.

    Br, Bt, Bz: 2D arrays
    """
    if sim.comm.rank == 0:
        print("Initializing magnetic field on the mesh...")

    iz_min = sim.comm.n_guard + sim.comm.nz_damp + sim.comm.n_inject
    iz_max = iz_min + Br.shape[0]

    # Save previous values on the grid, and replace them with the magnetic fields
    # (This is done in preparation for gathering among procs)
    saved_Br = sim.fld.interp[0].Br
    np.real(sim.fld.interp[0].Br)[ iz_min:iz_max, : ] = Br
    saved_Bt = sim.fld.interp[0].Bt
    np.real(sim.fld.interp[0].Bt)[ iz_min:iz_max, : ] = Bt
    saved_Bz = sim.fld.interp[0].Bz
    np.real(sim.fld.interp[0].Bz)[ iz_min:iz_max, : ] = Bz
    
    # Create a global field object across all subdomains, and copy the fields
    global_Nz, _ = sim.comm.get_Nz_and_iz(
                    local=False, with_damp=True, with_guard=False )
    global_zmin, global_zmax = sim.comm.get_zmin_zmax(
                    local=False, with_damp=True, with_guard=False )
    global_fld = Fields( global_Nz, global_zmax,
            sim.fld.Nr, sim.fld.rmax, sim.fld.Nm, sim.fld.dt,
            zmin=global_zmin, n_order=sim.fld.n_order, use_cuda=False)

    # Gather the fields of the interpolation grid
    for field in ['Br', 'Bt', 'Bz']:
        local_array = getattr( sim.fld.interp[0], field )
        gathered_array = sim.comm.gather_grid_array(
                            local_array, with_damp=True)
        setattr( global_fld.interp[0], field, gathered_array )

    # Now that the (gathered) fields are stored in global_fld,
    # copy the saved field back into the local grid
    sim.fld.interp[0].Br = saved_Br
    sim.fld.interp[0].Bt = saved_Bt
    sim.fld.interp[0].Bz = saved_Bz

    # Communicate the results from proc 0 to the other procs
    # and add it to the interpolation grid of sim.fld.
    # - First find the indices at which the fields should be added
    Nz_local, iz_start_local_domain = sim.comm.get_Nz_and_iz(
        local=True, with_damp=True, with_guard=False, rank=sim.comm.rank )
    _, iz_start_local_array = sim.comm.get_Nz_and_iz(
        local=True, with_damp=True, with_guard=True, rank=sim.comm.rank )
    iz_in_array = iz_start_local_domain - iz_start_local_array
    # - Then loop over modes and fields
    for field in ['Br', 'Bt', 'Bz']:
        # Get the local result from proc 0
        global_array = getattr( global_fld.interp[0], field )
        local_array = sim.comm.scatter_grid_array(
                                global_array, with_damp=True)
        # Add it to the fields of sim.fld
        local_field = getattr( sim.fld.interp[0], field )
        local_field[ iz_in_array:iz_in_array+Nz_local, : ] += local_array
    

    if sim.comm.rank == 0:
        print("Done.\n")