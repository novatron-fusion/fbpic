# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
from numba import vectorize, float64, void, njit
from scipy.constants import c
inv_c = 1./c
import numpy as np
import cupy as cp
import math
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import compile_cupy, cuda_tpb_bpg_1d
    from numba import cuda

class TrappedVelocities( object ):

    def __init__(self, Br, Bt, Bz, ptcl, Bmax,
                 Nz, Nr, dz, dr, zmin ):
        """
        Initialize a TrappedVelocities object, so that the function
        `field_func` can be called with `fieldtype` to generate trapped
        particle velocities

        Parameters
        ----------
        field_func: callable
            Function of the form `field_func( F, x, y, z, t, amplitude,
            length_scale )` and which returns the modified field F'
            (in the lab frame)

            This function will be called at each timestep, with:

            - F: 1d array of shape (N_ptcl,), containing the field
              designated by fieldtype, gathered on the particles
            - x, y, z: 1d arrays of shape (N_ptcl), containing the
              positions of the particles (in the lab frame)
            - t: float, the time in the simulation (in the lab frame)
            - amplitude and length_scale: floats that can be used within
              the function expression

            .. warning::
                In the PIC loop, this function is called after
                the field gathering. Thus this function can potentially
                overwrite the fields that were gathered on the grid. To avoid
                this, use "return(F + external_field) " inside the definition
                of `field_func` instead of "return(external_field)"

            .. warning::
                Inside the definition of `field_func` please use
                the `math` module for mathematical functions, instead of numpy.
                This will allow the function to be compiled for GPU.

        fieldtype: string
            Specifies on which field `field_func` will be applied.
            Either 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'

        ptcl: a list with Particles object
            The species on which the external field has to be applied.
        """
        # Register the arguments
        self.Br = cp.asarray(Br)
        self.Bt = cp.asarray(Bt)
        self.Bz = cp.asarray(Bz)
        self.ptcl = ptcl
        self.Bmax = Bmax

        self.Nz = Nz
        self.Nr = Nr
        self.dz = dz
        self.dr = dr
        self.zmin = zmin


    @compile_cupy
    def transform_cyl_to_cart_cuda( Fx, Fy, Fr, Ft, x, y ):
        i = cuda.grid(1)
        if i < Fx.shape[0]:
            r = math.sqrt(x[i]**2 + y[i]**2)
            if r > 0.:
                Fx[i] += Fr[i] * x[i] / r - Ft[i] * y[i] / r
                Fy[i] += Fr[i] * y[i] / r + Ft[i] * x[i] / r

    
    @compile_cupy
    def grid_data_bilinear_interp( F, field, x, y, z, dz, dr, Nz, Nr, zmin ):
        i = cuda.grid(1)
        if i < F.shape[0]:
            r = math.sqrt(x[i]**2 + y[i]**2)
            zi_min = math.floor((z[i] - zmin)  / dz)
            ri_min = math.floor(r / dr)

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

            fQ11 = field[int(zi_min + Nz*ri_min)]
            fQ12 = field[int(zi_min + Nz*(ri_min + 1))]
            fQ22 = field[int(zi_min + 1 + Nz*(ri_min + 1))]
            fQ21 = field[int(zi_min + 1 + Nz*ri_min)]

            b1 = fQ11 * (r2 - r) + fQ12 * (r - r1)
            b2 = fQ21 * (r2 - r) + fQ22 * (r - r1)

            F[i] += det * ((z2 - z[i]) * b1 + (z[i] - z1) * b2)
            
    """@compile_cupy
    def loss_cone( ux, uy, uz, Bx, By, Bz, Bmax):
        i = cuda.grid(1)
        if i < ux.shape[0]:
            Bmag = math.sqrt(Bx[i]**2 + By[i]**2 + Bz[i]**2)
            alpha_lc = math.asin(math.sqrt(Bmag / Bmax))

            Rm = Bmax / Bmag

            umag = math.sqrt(ux[i]**2 + uy[i]**2 + uz[i]**2)
            bx = Bx[i] / Bmag
            by = By[i] / Bmag
            bz = Bz[i] / Bmag

            u_dot_b = ux * bx + uy * by + uz * bz

            u_par_x = u_dot_b * bx / Bmag
            u_par_y = u_dot_b * by / Bmag
            u_par_z = u_dot_b * bz / Bmag

            u_perp_x = ux - u_par_x
            u_perp_y = uy - u_par_y
            u_perp_z = uz - u_par_z

            u_perp_mag = math.sqrt(u_perp_x**2 + u_perp_y**2 + u_perp_z**2)
            while u_perp_mag / umag > 1 / math.sqrt(Rm):
                U = xoroshiro128p_uniform_float64(
                random_states, i)"""





    def generate_trapped_velocities(self):
        """
        Gather external field function to the particles position
        and calculate the trapped velocity

        This function is called at each timestep, after field gathering
        in the step function.

        Parameters
        ----------
        ptcl: a list a Particles objects
            The particles on which the external fields will be applied
        """
        for species in self.ptcl:
            # Only apply the field if there are macroparticles
            # in this species
            if species.Ntot <= 0:
                print("species.Ntot <= 0")
                continue

            x = cp.asarray(species.x)
            y = cp.asarray(species.y)
            z = cp.asarray(species.z)

            ux = cp.asarray(species.ux)
            uy = cp.asarray(species.uy)
            uz = cp.asarray(species.uz)

            Bx = cp.zeros(species.Ntot)
            By = cp.zeros(species.Ntot)
            Br = cp.zeros(species.Ntot)
            Bt = cp.zeros(species.Ntot)

            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( species.Ntot )
            self.grid_data_bilinear_interp[dim_grid_1d, dim_block_1d]( 
                Br, self.Br, x, y, z,
                self.dz, self.dr, self.Nz, self.Nr, self.zmin)
            self.grid_data_bilinear_interp[dim_grid_1d, dim_block_1d]( 
                Bt, self.Bt, x, y, z,
                self.dz, self.dr, self.Nz, self.Nr, self.zmin)
            
            self.transform_cyl_to_cart_cuda[dim_grid_1d, dim_block_1d](
                        Bx, By, Br, Bt, x, y)
            
            Bz = cp.zeros(species.Ntot)
            self.grid_data_bilinear_interp[dim_grid_1d, dim_block_1d]( 
                Bz, self.Bz, x, y, z,
                self.dz, self.dr, self.Nz, self.Nr, self.zmin)
                

            Bmag = cp.sqrt(Bx**2 + By**2 + Bz**2)
            alpha_lc = cp.arcsin(cp.sqrt(Bmag / self.Bmax))

            todo = cp.ones(x.shape, dtype=bool)
            umag = cp.zeros_like(x)
            alpha = cp.zeros_like(x)

            while True:
                umag[todo] = cp.sqrt(ux[todo]**2 + uy[todo]**2 + uz[todo]**2) 
                alpha[todo] = cp.arccos((ux[todo] * Bx[todo] + uy[todo] * By[todo] + uz[todo] * Bz[todo]) / (umag[todo] * Bmag[todo]))

                alpha[todo] = cp.where(alpha[todo] < np.pi / 2, alpha[todo], np.pi - alpha[todo])
                todo = cp.where(cp.abs(alpha) < cp.abs(alpha_lc))[0]

                print(todo.shape[0])

                if not todo.any():
                    break
                    
                # Generate a new random vector
                u_new = cp.random.randn(todo.shape[0], 3)

                # Calculate its magnitude
                mag_u_new = cp.linalg.norm(u_new, axis=1, keepdims=True)

                # Normalize
                u_new = u_new / mag_u_new

                # Scale the new vector to have the same magnitude as the original vector
                ux[todo] = umag[todo] * u_new[:, 0]
                uy[todo] = umag[todo] * u_new[:, 1]
                uz[todo] = umag[todo] * u_new[:, 2]

            species.ux = cp.asnumpy(ux)
            species.uy = cp.asnumpy(uy)
            species.uz = cp.asnumpy(uz)





