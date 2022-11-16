# Copyright 2020, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Alberto de la Ossa
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the Perfectly Electric Conductor (PEC) class, which set the 
fields to 0 inside the wall
"""
from scipy.constants import c
import math as m
import numpy as np
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_2d, compile_cupy


@cuda.jit( device=True, inline=True )
def ray_casting(z, r, polygon):
    n = len(polygon)
    count = 0
    for i in range(n-1):
        z1 = polygon[i][0]
        z2 = polygon[i+1][0]
        r1 = polygon[i][1]
        r2 = polygon[i+1][1]

        if (r < r1) != (r < r2) \
            and z < (z2-z1) * (r-r1) / (r2-r1) + z1:
            count += 1
        
    return(False if count % 2 == 0 else True)


@cuda.jit( device=True, inline=True )
def grid_data_bilinear_interp( field, z, r, dz, dr, Nz, Nr, zmin):
        zi_min = int(m.floor((z - zmin)  / dz))
        ri_min = int(m.floor(r / dr))

        if zi_min >= Nz-1:
            zi_min = Nz-2
        elif zi_min < 0:
            zi_min = 0
        if ri_min >= Nr-1:
            ri_min = Nr-2
        
        r1 = dr * ri_min
        z1 = zmin + dz * zi_min 
        r2 = r1 + dr
        z2 = z1 + dz
        det = 1. / ((z2 - z1) * (r2 - r1))

        fQ11 = field[zi_min, ri_min]
        fQ12 = field[zi_min, ri_min+1]
        fQ22 = field[zi_min+1, ri_min+1]
        fQ21 = field[zi_min+1, ri_min]

        b1 = fQ11 * (r2 - r) + fQ12 * (r - r1)
        b2 = fQ21 * (r2 - r) + fQ22 * (r - r1)

        return (det * ((z2 - z) * b1 + (z - z1) * b2))


@cuda.jit( device=True, inline=True )
def P00( x, h, L ):
    return (-(x - h)*(x + L)**2 / ( h * L**2 ) )


@cuda.jit( device=True, inline=True )
def P01( x, h, L ):
    return ( x*(x + L)**2 / ( h * (h + L)**2 ) )


@cuda.jit( device=True, inline=True )
def ray_casting(z, r, polygon):
    n = len(polygon)
    count = 0
    for i in range(n-1):
        z1 = polygon[i][0]
        z2 = polygon[i+1][0]
        r1 = polygon[i][1]
        r2 = polygon[i+1][1]

        if (r < r1) != (r < r2) \
            and z < (z2-z1) * (r-r1) / (r2-r1) + z1:
            count += 1
        
    return(False if count % 2 == 0 else True)


class PEC(object):

    def __init__( self, wall_arr, upper_segment, lower_segment, normal, tangent, side, z_start=None, h=None, L=None, gamma_boost=None, m='all'):
        """
        Initialize a perfect electric conductor.

        The wall reflects the fields

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
        self.side = side

        # Calculate array of normal vectors
        self.normal = cupy.asarray(normal)
        self.tangent = cupy.asarray(tangent)
        self.upper_segment = cupy.asarray(upper_segment)
        self.lower_segment = cupy.asarray(lower_segment)
        
        self.z_start = z_start
        self.h = h
        self.L = L
        # static, active, mirror
        self.method = 'static'
        self.gamma_boost = gamma_boost
        if m == 'all':
            self.modes = None
        elif isinstance(m, int):
            self.modes = [m]
        elif isinstance(m, list):
            self.modes = m
        else:
            raise TypeError('m should be an int or a list of ints.')


    def construct_penalty_term( self, interp, comm, iteration ):
        """
        Construct penalty term for PEC boundary condition

        Parameters:
        -----------
        interp: a list of InterpolationGrid objects
            Contains the values of the fields in interpolation space
        """

        # Calculate indices in z between which the field should be set to 0
        zmin_global, zmax_global = comm.get_zmin_zmax( local=False,
                        with_guard=True, with_damp=True, rank=comm.rank)

        z = cupy.linspace(zmin_global, zmax_global, interp[0].Nz)
        r = cupy.linspace(0., interp[0].rmax, interp[0].Nr)

        for i, grid in enumerate(interp):
            if self.modes is not None:
                if i not in self.modes:
                    continue
            
            if iteration == 0:
                self.s = cupy.empty((interp[0].Nz, interp[0].Nr), dtype=np.float64)
                dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(interp[0].Nz, interp[0].Nr)
                self.calculate_local_coord_syst[dim_grid_2d, dim_block_2d]( self.s, 
                    z, r, 
                    self.wall_arr,
                    self.upper_segment,
                    self.lower_segment,
                    self.normal,
                    self.tangent,
                    interp[0].rmax)
                self.segments = cupy.zeros((interp[0].Nz, interp[0].Nr), dtype=np.int32)
                self.calculate_segments[dim_grid_2d, dim_block_2d]( self.segments, 
                    z, r, 
                    self.wall_arr,
                    self.upper_segment,
                    self.lower_segment)
                    
            Er = getattr( grid, 'Er')
            Et = getattr( grid, 'Et')
            Ez = getattr( grid, 'Ez')

            self.Er_old = Er
            self.Et_old = Et
            self.Ez_old = Ez
                    

    def reflect_fields_at_pec( self, interp, comm, iteration ):
        """
        Reflect fields at perfect electric conductor
        """
        if self.z_start != None:

            # Calculate indices in z between which the field should be set to 0
            zmin_global, zmax_global = comm.get_zmin_zmax( local=False,
                            with_guard=True, with_damp=True, rank=comm.rank)

            z = cupy.linspace(zmin_global, zmax_global, interp[0].Nz)
            r = cupy.linspace(0., interp[0].rmax, interp[0].Nr)

            """
            for i, grid in enumerate(interp):
                if self.modes is not None:
                    if i not in self.modes:
                        continue
                Er = getattr( grid, 'Er')
                Et = getattr( grid, 'Et')
                Ez = getattr( grid, 'Ez')
                
                if iteration % 1000 == 0:
                    self.complexDTVFilter(Ez, 15, 5, z, r, self.upper_segment, self.lower_segment, self.normal)
                    self.complexDTVFilter(Er, 15, 5, z, r, self.upper_segment, self.lower_segment, self.normal)
                    self.complexDTVFilter(Et, 15, 5, z, r, self.upper_segment, self.lower_segment, self.normal)

                if self.side == 'left':
                    eta = (1 / 1)
                    Er[iz-3:iz,:] = Er[iz-3:iz,:] - eta * self.Er_old[iz-3:iz,:]
                    Et[iz-3:iz,:] = Et[iz-3:iz,:] - eta * self.Et_old[iz-3:iz,:]
                    Ez[iz-3:iz,:] = Ez[iz-3:iz,:] - eta * self.Ez_old[iz-3:iz,:]

                else:
                    eta = (1 / 1)
                    Er[iz:iz+3,:] = Er[iz:iz+3,:] - eta * self.Er_old[iz:iz+3,:]
                    Et[iz:iz+3,:] = Et[iz:iz+3,:] - eta * self.Et_old[iz:iz+3,:]
                    Ez[iz:iz+3,:] = Ez[iz:iz+3,:] - eta * self.Ez_old[iz:iz+3,:]

                    #Er[imin:imax,:] = Er[imin:imax,:] - self.penalty_term_r
                    #Et[imin:imax,:] = Et[imin:imax,:] - self.penalty_term_t
                    #Ez[imin:imax,:] = Ez[imin:imax,:] - self.penalty_term_z
            """

            
            for i, grid in enumerate(interp):
                if self.modes is not None:
                    if i not in self.modes:
                        continue

                Er = getattr( grid, 'Er')
                Et = getattr( grid, 'Et')
                Ez = getattr( grid, 'Ez')
                #Br = getattr( grid, 'Br')
                #Bt = getattr( grid, 'Bt')
                #Bz = getattr( grid, 'Bz')

                iz = int((0. - zmin_global) / interp[0].dz)

                dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(interp[0].Nz, interp[0].Nr)
                if self.method == 'active':
                    self.pec_lower_penalty_term[dim_grid_2d, dim_block_2d]( Ez, Er, Et,
                                                                self.Ez_old, self.Er_old, self.Et_old,
                                                                z, r, self.wall_arr,
                                                                self.s, self.h, self.L,
                                                                self.lower_segment,
                                                                self.normal,
                                                                interp[0].dz, interp[0].dr,
                                                                interp[0].Nz, interp[0].Nr,
                                                                interp[0].rmax,
                                                                zmin_global)
                    self.pec_upper_penalty_term[dim_grid_2d, dim_block_2d]( Ez, Er, Et,
                                                                self.Ez_old, self.Er_old, self.Et_old,
                                                                z, r, self.wall_arr,
                                                                self.s, self.h, self.L,
                                                                self.upper_segment,
                                                                self.lower_segment,
                                                                self.normal,
                                                                interp[0].dz, interp[0].dr,
                                                                interp[0].Nz, interp[0].Nr,
                                                                interp[0].rmax,
                                                                zmin_global)
                elif self.method == 'mirror':
                    """
                    self.pec_lower_penalty_term[dim_grid_2d, dim_block_2d]( Ez, Er, Et,
                                                                self.Ez_old, self.Er_old, self.Et_old,
                                                                z, r, self.wall_arr,
                                                                self.s, self.h, self.L,
                                                                self.lower_segment,
                                                                self.normal,
                                                                interp[0].dz, interp[0].dr,
                                                                interp[0].Nz, interp[0].Nr,
                                                                interp[0].rmax,
                                                                zmin_global)
                    """
                    self.pec_mirror_image[dim_grid_2d, dim_block_2d]( Ez, Er, Et,
                                            self.Ez_old, self.Er_old, self.Et_old,
                                            z, r, self.wall_arr,
                                            self.upper_segment,
                                            self.normal,
                                            self.tangent,
                                            self.s, self.h, self.L,
                                            interp[0].dz, interp[0].dr,
                                            interp[0].Nz, interp[0].Nr,
                                            interp[0].rmax,
                                            zmin_global)
                else:
                    self.pec_static_penalty_term[dim_grid_2d, dim_block_2d]( Ez, Er, Et,
                                                                        self.Ez_old, self.Er_old, self.Et_old,
                                                                        z, r, self.segments,
                                                                        interp[0].rmax)
                
                if iteration % 100 == 0:
                    self.complexDTVFilter(Ez, 9, 2, z, r, self.segments, self.normal)
                    self.complexDTVFilter(Er, 9, 2, z, r, self.segments, self.normal)
                    self.complexDTVFilter(Et, 9, 2, z, r, self.segments, self.normal)
                    #self.complexDTVFilter(Bz, 9, 2, z, r, self.segments, self.normal)
                    #self.complexDTVFilter(Br, 9, 2, z, r, self.segments, self.normal)
                    #self.complexDTVFilter(Bt, 9, 2, z, r, self.segments, self.normal)
                
    @compile_cupy
    def pec_mirror_image( Ez, Er, Et,
                    Ez_o, Er_o, Et_o,
                    z, r, wall_arr,
                    upper_segment,
                    normal,
                    tangent,
                    s, h, L,
                    dz, dr, 
                    Nz, Nr,
                    rmax,
                    zmin):
        i, j = cuda.grid(2)
        if i < Ez.shape[0] and j < Ez.shape[1]:
            if r[j] < rmax and r[j] > 0.:
                in_poly = ray_casting(z[i], r[j], wall_arr)
                if not in_poly:
                    n = len(wall_arr)
                    for k in range(n-1):
                        in_upper_segment = ray_casting(z[i], r[j], upper_segment[k])
                        if in_upper_segment:
                            """
                            z1 = upper_segment[k][0,0]
                            r1 = upper_segment[k][0,1]

                            z2 = z[i]
                            r2 = r[j]

                            det_inv = 1. / (normal[k,0]*(-tangent[k,1]) - (-tangent[k,0])*normal[k,1])
                            if det_inv == 0:
                                break

                            Ainv_11 = det_inv * (-tangent[k,1])
                            Ainv_12 = det_inv * (tangent[k,0])

                            a = Ainv_11 * (z2 - z1) + Ainv_12 * (r2 - r1)

                            l1z = z2 + a * normal[k,0]
                            l1r = r2 + a * normal[k,1]

                            a_m = z2 - 2*(l1z - z2)
                            b_m = r2 - 2*(l1r - r2)

                            Ez_m = grid_data_bilinear_interp( Ez, a_m, b_m, dz, dr, Nz, Nr, zmin)
                            Er_m = grid_data_bilinear_interp( Er, a_m, b_m, dz, dr, Nz, Nr, zmin)
                            Et_m = grid_data_bilinear_interp( Et, a_m, b_m, dz, dr, Nz, Nr, zmin)

                            E_dot_n = (Ez_m*normal[k,0] + Er_m*normal[k,1] + Et_m*normal[k,2])
                            Ez_perp = E_dot_n * normal[k,0]
                            Er_perp = E_dot_n * normal[k,1]
                            Et_perp = E_dot_n * normal[k,2]
                
                            n_cross_E0 = (Et[i,j]*normal[k,1]-Er[i,j]*normal[k,2])
                            n_cross_E1 = -(Et[i,j]*normal[k,0]-Ez[i,j]*normal[k,2])
                            n_cross_E2 = (Er[i,j]*normal[k,0]-Ez[i,j]*normal[k,1])

                            Ez_par = -(n_cross_E2*normal[k,1] - n_cross_E1*normal[k,2])
                            Er_par = (n_cross_E2*normal[k,0] - n_cross_E0*normal[k,2])
                            Et_par = -(n_cross_E1*normal[k,0] - n_cross_E0*normal[k,1])

                            #Ez[i,j] = Ez_perp - Ez_par
                            #Er[i,j] = Er_perp - Er_par
                            #Et[i,j] = Et_perp - Et_par
                            """
                            damp = P00(-s[i,j], h, L) + P01(-s[i,j], h, L)

                            Ez[i,j] *= damp
                            Er[i,j] *= damp
                            Et[i,j] *= damp
                            break
                        else:
                            eta = 0.9
                            Ez[i,j] = Ez[i,j] - eta * Ez_o[i,j]
                            Er[i,j] = Er[i,j] - eta * Er_o[i,j]
                            Et[i,j] = Et[i,j] - eta * Et_o[i,j]

    @compile_cupy
    def calculate_local_coord_syst( s, z, r, 
                    wall_arr,
                    upper_segment,
                    lower_segment,
                    normal,
                    tangent,
                    rmax):
        i, j = cuda.grid(2)
        if i < s.shape[0] and j < s.shape[1]:
            if r[j] < rmax and r[j] > 0.:
                n = len(wall_arr)
                for k in range(n-1):
                    in_lower_segment = ray_casting(z[i], r[j], lower_segment[k])
                    in_upper_segment = ray_casting(z[i], r[j], upper_segment[k])
                    if in_lower_segment or in_upper_segment:
                        if lower_segment[k][0,1] == rmax and lower_segment[k][1,1] == rmax:
                            break
                        if in_lower_segment:
                            z1 = lower_segment[k][0,0]
                            r1 = lower_segment[k][0,1]
                        else:
                            z1 = upper_segment[k][0,0]
                            r1 = upper_segment[k][0,1]
                        
                        z2 = z[i]
                        r2 = r[j]

                        det_inv = 1. / (normal[k,0]*(-tangent[k,1]) - (-tangent[k,0])*normal[k,1])
                        if det_inv == 0:
                            break

                        Ainv_11 = det_inv * (-tangent[k,1])
                        Ainv_12 = det_inv * (tangent[k,0])

                        s[i,j] = Ainv_11 * (z2 - z1) + Ainv_12 * (r2 - r1)
                        break

    @compile_cupy
    def calculate_segments( segments, z, r, 
                    wall_arr,
                    upper_segment,
                    lower_segment):
        i, j = cuda.grid(2)
        if i < segments.shape[0] and j < segments.shape[1]:
            n = len(wall_arr)
            for k in range(n-1):
                in_lower_segment = ray_casting(z[i], r[j], lower_segment[k])
                in_upper_segment = ray_casting(z[i], r[j], upper_segment[k])
                in_poly = ray_casting(z[i], r[j], wall_arr)
                if in_lower_segment:
                    segments[i,j] = 1
                    break
                elif in_upper_segment:
                    segments[i,j] = 2
                    break
                elif not in_poly:
                    segments[i,j] = 3
                    break

    @compile_cupy
    def pec_lower_penalty_term( Ez, Er, Et,
                    Ez_o, Er_o, Et_o,
                    z, r, wall_arr,
                    s, h, L,
                    lower_segment,
                    normal,
                    dz, dr, 
                    Nz, Nr,
                    rmax,
                    zmin):
        i, j = cuda.grid(2)
        if i < Ez.shape[0] and j < Ez.shape[1]:
            if r[j] < rmax and r[j] > 0.:
                in_poly = ray_casting(z[i], r[j], wall_arr)
                found = False
                n = len(wall_arr)
                if in_poly:
                    for k in range(n-1):
                        in_lower_segment = ray_casting(z[i], r[j], lower_segment[k])
                        
                        if in_lower_segment:
                            if lower_segment[k][0,1] == rmax and lower_segment[k][1,1] == rmax:
                                break
                            
                            z2 = z[i]
                            r2 = r[j]
                            
                            a_s = z2 - s[i,j] * normal[k,0]
                            b_s = r2 - s[i,j] * normal[k,1]

                            a_h = a_s - h * normal[k,0]
                            b_h = b_s - h * normal[k,1]
                            
                            Ez_s = grid_data_bilinear_interp( Ez, a_s, b_s, dz, dr, Nz, Nr, zmin)
                            Er_s = grid_data_bilinear_interp( Er, a_s, b_s, dz, dr, Nz, Nr, zmin)
                            Et_s = grid_data_bilinear_interp( Et, a_s, b_s, dz, dr, Nz, Nr, zmin)

                            Ez_h = grid_data_bilinear_interp( Ez, a_h, b_h, dz, dr, Nz, Nr, zmin)
                            Er_h = grid_data_bilinear_interp( Er, a_h, b_h, dz, dr, Nz, Nr, zmin)
                            Et_h = grid_data_bilinear_interp( Et, a_h, b_h, dz, dr, Nz, Nr, zmin)

                            E_dot_n = (Ez_s*normal[k,0] + Er_s*normal[k,1] + Et_s*normal[k,2])
                            Ez_perp = E_dot_n * normal[k,0]
                            Er_perp = E_dot_n * normal[k,1]
                            Et_perp = E_dot_n * normal[k,2]

                            p0 = P00(-s[i,j], h, L)
                            p1 = P01(-s[i,j], h, L)

                            gz = Ez_perp * p0 + Ez_h * p1
                            gr = Er_perp * p0 + Er_h * p1
                            gt = Et_perp * p0 + Et_h * p1

                            eta = 0.25
                            Ez[i,j] = Ez[i,j] - eta * (Ez_o[i,j] - gz)
                            Er[i,j] = Er[i,j] - eta * (Er_o[i,j] - gr)
                            Et[i,j] = Et[i,j] - eta * (Et_o[i,j] - gt)
                            break

    @compile_cupy
    def pec_upper_penalty_term( Ez, Er, Et,
                    Ez_o, Er_o, Et_o,
                    z, r, wall_arr,
                    s, h, L,
                    upper_segment,
                    lower_segment,
                    normal,
                    dz, dr, 
                    Nz, Nr,
                    rmax,
                    zmin):
        i, j = cuda.grid(2)
        if i < Ez.shape[0] and j < Ez.shape[1]:
            if r[j] < rmax and r[j] > 0.:
                in_poly = ray_casting(z[i], r[j], wall_arr)
                found = False
                n = len(wall_arr)
                for k in range(n-1):
                    in_upper_segment = ray_casting(z[i], r[j], upper_segment[k])
                    in_lower_segment = ray_casting(z[i], r[j], lower_segment[k])
                    if in_upper_segment and not in_lower_segment and not in_poly:
                    
                        z2 = z[i]
                        r2 = r[j]
                        
                        a_s = z2 - s[i,j] * normal[k,0]
                        b_s = r2 - s[i,j] * normal[k,1]

                        a_h = a_s - h * normal[k,0]
                        b_h = b_s - h * normal[k,1]

                        Ez_s = grid_data_bilinear_interp( Ez_o, a_s, b_s, dz, dr, Nz, Nr, zmin)
                        Er_s = grid_data_bilinear_interp( Er_o, a_s, b_s, dz, dr, Nz, Nr, zmin)
                        Et_s = grid_data_bilinear_interp( Et_o, a_s, b_s, dz, dr, Nz, Nr, zmin)

                        Ez_h = grid_data_bilinear_interp( Ez_o, a_h, b_h, dz, dr, Nz, Nr, zmin)
                        Er_h = grid_data_bilinear_interp( Er_o, a_h, b_h, dz, dr, Nz, Nr, zmin)
                        Et_h = grid_data_bilinear_interp( Et_o, a_h, b_h, dz, dr, Nz, Nr, zmin)

                        E_dot_n = (Ez_s*normal[k,0] + Er_s*normal[k,1] + Et_s*normal[k,2])
                        Ez_perp = E_dot_n * normal[k,0]
                        Er_perp = E_dot_n * normal[k,1]
                        Et_perp = E_dot_n * normal[k,2]

                        p0 = P00(-s[i,j], h, L)
                        p1 = P01(-s[i,j], h, L)

                        gz = Ez_perp * p0 + Ez_h * p1
                        gr = Er_perp * p0 + Er_h * p1
                        gt = Et_perp * p0 + Et_h * p1

                        eta = 0.25
                        Ez[i,j] = Ez[i,j] - eta * (Ez_o[i,j] - gz)
                        Er[i,j] = Er[i,j] - eta * (Er_o[i,j] - gr)
                        Et[i,j] = Et[i,j] - eta * (Et_o[i,j] - gt)
                        found = True
                        break
                if not in_poly and found == False:
                    Ez[i,j] = Ez[i,j] - 0.90*Ez_o[i,j]
                    Er[i,j] = Er[i,j] - 0.90*Er_o[i,j]
                    Et[i,j] = Et[i,j] - 0.90*Et_o[i,j]

    @compile_cupy
    def pec_static_penalty_term( Ez, Er, Et,
                    Ez_o, Er_o, Et_o,
                    z, r, segments,
                    rmax):
        i, j = cuda.grid(2)
        if i < Ez.shape[0] and j < Ez.shape[1]:
            if r[j] < rmax and r[j] > 0.:
                # segments: 2 upper, segments: 3 otuside polygon
                if segments[i,j] == 2 or segments[i,j] == 3:
                    Ez[i,j] = Ez[i,j] - 0.90*Ez_o[i,j]
                    Er[i,j] = Er[i,j] - 0.90*Er_o[i,j]
                    Et[i,j] = Et[i,j] - 0.90*Et_o[i,j]


    @compile_cupy
    def DTVStep(u0, U, v, a, dt, Lambda, z, r, segments, normal):
        i, j = cuda.grid(2)
        N = u0.shape[0]
        M = u0.shape[1]
        if i > 0 and j > 0 and i < N-1 and j < M-1:
            u1 = U[i,j]
            u1_inv = 1 / u1
            tol = 1.e-3
            if abs((u1 - U[i+1,j]) * u1_inv) > tol \
                or abs((u1 - U[i-1,j]) * u1_inv) > tol \
                or abs((u1 - U[i,j+1]) * u1_inv) > tol \
                or abs((u1 - U[i,j-1]) * u1_inv) > tol:
                if segments[i,j] == 1 or segments[i,j] == 2:
                    Lambda = 15
                else:
                    Lambda = 9

                s = ( (U[i,j-1]-u1)**2 + (U[i,j+1]-u1)**2 \
                        + (U[i-1,j]-u1)**2 + (U[i+1,j]-u1)**2 + a )**0.5

                sim = ( (U[i-1,j-1]-u1)**2 + (U[i-1,j+1]-U[i-1,j])**2 \
                        + (U[i-2,j]-U[i-1,j])**2 + (u1-U[i-1,j])**2 + a )**0.5

                sip = ( (U[i+1,j-1]-U[i+1,j])**2 + (U[i+1,j+1]-U[i+1,j])**2 \
                        + (u1-U[i+1,j])**2 + (U[i+2,j]-U[i+1,j])**2 + a )**0.5

                sjm = ( (U[i,j-2]-U[i,j-1])**2 + (u1-U[i,j-1])**2 \
                        + (U[i-1,j-1]-U[i,j-1])**2 + (U[i+1,j-1]-U[i,j-1])**2 + a )**0.5

                sjp = ( (u1-U[i,j+1])**2 + (U[i,j+2]-U[i,j+1])**2 \
                        + (U[i-1,j+1]-U[i,j+1])**2 + (U[i+1,j+1]-U[i,j+1])**2 + a )**0.5
                """
                #i=0 j=0
                s[0,0] = ( (U[0,1]-U[0,0])**2 + (U[1,0]-U[0,0])**2 + a )**0.5

                #i=N-1; j=M-1
                s[N-1,M-1] = ( (U[N-1,M-2]-U[N-1,M-1])**2 + (U[N-2,M-1]-U[N-1,M-1])**2 )**0.5

                #i=N-1; j=0
                s[N-1,0] = ( (U[N-1,1]-U[N-1,0])**2 + (U[N-2,0]-U[N-1,0])**2 )**0.5

                #i=0; j=M-1
                s[0,M-1] = ( (U[0,M-2]-U[0,M-1])**2 + (U[1,M-1]-U[0,M-1])**2 + a )**0.5
                """
                
                v[i,j] = u1 + dt*( ( U[i+1,j] - u1 ) * ( 1. + s / sip ) + 
                                ( U[i-1,j] - u1 ) * ( 1. + s / sim ) +
                                ( U[i,j+1] - u1 ) * ( 1. + s / sjp ) +
                                ( U[i,j-1] - u1 ) * ( 1. + s / sjm) -
                                Lambda * s * ( u1 - u0[i,j] ) )


    def complexDTVFilter(self, u0, Lambda, timeSteps, 
                        z, r, segments, normal):
        a = 1e-6
        a = a**2
        N = u0.shape[0]
        M = u0.shape[1]
        dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(N, M)
        
        dt = 0.002

        v = u0.real
        U = u0.real
        k = 1
        while k <= timeSteps:
            self.DTVStep[dim_grid_2d, dim_block_2d](u0.real, U, v, 
                a, dt, Lambda, z, r, segments, normal)

            k = k+1
            U = v
        u0.real = U
        
        v = u0.imag
        U = u0.imag
        k = 1
        while k <= timeSteps:
            self.DTVStep[dim_grid_2d, dim_block_2d](u0.imag, U, v, 
                a, dt, Lambda, z, r, segments, normal)

            k = k+1
            U = v
        u0.imag = U
        

