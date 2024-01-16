# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
from numba import vectorize, float64, void, njit
from scipy.constants import c
inv_c = 1./c
import numpy as np
import cupy
import math
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import compile_cupy, cuda_tpb_bpg_1d
    from numba import cuda

class ExternalField( object ):

    def __init__(self, field_func, fieldtype, amplitude,
                 length_scale, species=None, Bmag=None, gamma_boost=None,
                 Nz=None, Nr=None, dz=None, dr=None):
        """
        Initialize an ExternalField object, so that the function
        `field_func` is called at each time step on the field `fieldtype`

        This object should be added to the list `external_fields`,
        which is an attribute of the Simulation object, so that the
        fields are applied at each timestep. (See the example below)

        The function `field_func` is automatically converted to a GPU
        function if needed, by using numba's ufunc feature.

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

        species: a Particles object, optionals
            The species on which the external field has to be applied.
            If no species is specified, the external field is applied
            to all particles.

        gamma_boost: float, optional
            When running the simulation in a boosted frame, set the
            value of `gamma_boost` to the corresponding Lorentz factor.
            The external fields will be automatically converted to the
            boosted frame.

        Example
        -------
        In order to define a magnetic undulator, polarized along y, with
        a field of 1 Tesla and a period of 1 cm :

        ::

            def field_func( F, x, y, z, t , amplitude, length_scale ):
                return( F + amplitude * math.cos( 2*np.pi*z/length_scale ) )

            sim.external_fields = [ ExternalField( field_func, 'By', 1., 1.e-2 ) ]

        .. warning::

            Note that, in principle, ``field_func`` does not necessarily need
            to use the arguments ``amplitude`` and ``length_scale``.
            For instance, in the above example, we could have used

            ::

                def field_func( F, x, y, z, t , amplitude, length_scale ):
                    return( F + 1. * math.cos( 2*np.pi*z/1.e-2 ) )

            However, **when running the simulation in a boosted frame**
            (i.e. when setting the above argument ``gamma_boost``),
            the expression of the external fields **needs to be proportional
            to** ``amplitude``. This is because, internally, the automatic
            conversion of the external fields to the boosted frame relies
            on this variable. (There is no similar constraint for
            ``length_scale``, however.)
        """
        # Register the arguments
        self.length_scale = length_scale
        self.species = species
        self.Bmag = Bmag

        self.Nz = Nz
        self.Nr = Nr
        self.dz = dz
        self.dr = dr
        # Check that fieldtype is a correct field
        if (fieldtype in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Br', 'Bt']) is False:
            raise ValueError("`fieldtype` must be one of Ex, Ey, Ez, Bx, By, Bz, Br, Bt")

        if type(field_func) is np.ndarray:
            self.field_func = field_func
            if cuda_installed:
                self.field_func_d = cupy.asarray(field_func)
        else:
            # Note: when `gamma_boost` is passed, the fields are evaluated in
            # the boosted frame, even though the user-provided function corresponds
            # to the lab frame. This is done (conceptually) in 2 steps:
            # - the field is computed in the lab frame, by transforming
            #   the current position/time of particles to the lab frame
            # - the field is converted back to the boosted frame, by using
            #   Lorentz transform formulas for E and B

            # Modify user-input function, so as to evaluate field in the lab frame
            if (gamma_boost is not None) and (gamma_boost != 1.):
                beta_boost = np.sqrt(1. - 1./gamma_boost**2)
                field_func = njit(field_func)
                def func( F, x, y, z, t, amplitude, length_scale ):
                    zlab = gamma_boost*(z + beta_boost*c*t)
                    tlab = gamma_boost*(t + beta_boost*inv_c*z)
                    return field_func(F, x, y, zlab, tlab, amplitude, length_scale)
            else:
                func = field_func
            self.field_func = field_func

            # Compile the field_func for cpu and gpu
            signature = [ float64( float64, float64, float64,
                                float64, float64, float64, float64 ) ]
            cpu_compiler = vectorize( signature, target='cpu', nopython=True )
            self.cpu_func = cpu_compiler( func )
            if cuda_installed:
                # First create a device inline function
                inline_func = cuda.jit( func, inline=True, device=True )
                # Then create a CUDA kernel and compile it the usual way
                def external_field_kernel( F, x, y, z, t, amplitude, length_scale ):
                    i = cuda.grid(1)
        
                    if i < F.shape[0]:
                        F[i] = inline_func( F[i], x[i], y[i], z[i], t, amplitude, length_scale )

                # To ensure that the kernel is compiled immediately and prevent scoping issues,
                # it is specialized using an explicit signature
                gpu_signature = void( float64[:], float64[:], float64[:],
                                float64[:], float64, float64, float64 )

                self.gpu_func = compile_cupy( external_field_kernel ).specialize( gpu_signature )

        # Convert the field back to the boosted frame
        if (gamma_boost is not None) and (gamma_boost != 1.):
            g = gamma_boost
            gb = gamma_boost*beta_boost
            if fieldtype == 'Ex':
                self.fieldtypes_and_amplitudes = (('Ex', g*amplitude),
                                                ('By', -gb*inv_c*amplitude))
            elif fieldtype == 'Ey':
                self.fieldtypes_and_amplitudes = (('Ey', g*amplitude),
                                                ('Bx', gb*inv_c*amplitude))
            elif fieldtype == 'Bx':
                self.fieldtypes_and_amplitudes = (('Bx', g*amplitude),
                                                ('Ey', gb*c*amplitude))
            elif fieldtype == 'By':
                self.fieldtypes_and_amplitudes = (('By', g*amplitude),
                                                ('Ex', -gb*c*amplitude))
            elif (fieldtype == 'Ez') or (fieldtype == 'Bz'):
                self.fieldtypes_and_amplitudes = ((fieldtype, amplitude),)
        else:
            self.fieldtypes_and_amplitudes = ((fieldtype, amplitude),)


    @compile_cupy
    def transform_cyl_to_cart_cuda( Fx, Fy, Fr, Ft, x, y ):
        i = cuda.grid(1)
        if i < Fx.shape[0]:
            r = math.sqrt(x[i]**2 + y[i]**2)
            if r > 0.:
                Fx[i] += Fr[i] * x[i] / r - Ft[i] * y[i] / r
                Fy[i] += Fr[i] * y[i] / r + Ft[i] * x[i] / r

    
    @compile_cupy
    def grid_data_bilinear_interp( F, field, x, y, z, dz, dr, Nz, Nr, zmin):
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
            

    def apply_expression( self, ptcl, t, comm ):
        """
        Apply the external field function to the particles

        This function is called at each timestep, after field gathering
        in the step function.

        Parameters
        ----------
        ptcl: a list a Particles objects
            The particles on which the external fields will be applied

        t: float (seconds)
            The time in the simulation
        """
        for species in ptcl:

            # If any species was specified at initialization,
            # apply the field only on this species
            if (self.species is None) or (species is self.species):

                # Only apply the field if there are macroparticles
                # in this species
                if species.Ntot <= 0:
                    continue
                
                # Loop over the different fields involved
                for (fieldtype, amplitude) in self.fieldtypes_and_amplitudes:
                    if fieldtype == 'Br':
                        Bx = getattr( species, 'Bx' )
                        By = getattr( species, 'By' )
                        Br = cupy.zeros(species.Ntot)
                        Bt = cupy.zeros(species.Ntot)
                        if type(self.field_func_d) is cupy.ndarray:
                            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( species.Ntot )
                            self.grid_data_bilinear_interp[dim_grid_1d, dim_block_1d]( 
                                Br, self.field_func_d,
                                species.x, species.y, species.z,
                                self.dz, self.dr, self.Nz, self.Nr,
                                comm._zmin_global_domain )
                            self.grid_data_bilinear_interp[dim_grid_1d, dim_block_1d]( 
                                Bt, self.field_func_d,
                                species.x, species.y, species.z,
                                self.dz, self.dr, self.Nz, self.Nr,
                                comm._zmin_global_domain )
                            self.transform_cyl_to_cart_cuda[dim_grid_1d, dim_block_1d](
                                Bx, By, Br, Bt, species.x, species.y )
                        else:
                            # Get the threads per block and the blocks per grid
                            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( species.Ntot )
                            # Call the GPU kernel
                            self.gpu_func[dim_grid_1d, dim_block_1d](
                                Br, species.x, species.y, species.z,
                                t, amplitude, self.length_scale )
                            
                            # Call the GPU kernel
                            self.gpu_func[dim_grid_1d, dim_block_1d](
                                Bt, species.x, species.y, species.z,
                                t, amplitude, self.length_scale )

                            self.transform_cyl_to_cart_cuda[dim_grid_1d, dim_block_1d](
                                Bx, By, Br, Bt, species.x, species.y )
                    elif fieldtype == 'Bt':
                        continue
                    else:
                        field = getattr( species, fieldtype )
                        if type( field ) is np.ndarray:
                            # Call the CPU function
                            self.cpu_func( field, species.x, species.y, species.z,
                                t, amplitude, self.length_scale, out=field )
                        else:
                            # Get the threads per block and the blocks per grid
                            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( species.Ntot )
                            if type( self.field_func ) is np.ndarray:
                                self.grid_data_bilinear_interp[dim_grid_1d, dim_block_1d]( 
                                    field, self.field_func_d,
                                    species.x, species.y, species.z,
                                    self.dz, self.dr, self.Nz, self.Nr,
                                    comm._zmin_global_domain )
                            else:
                                # Call the GPU kernel
                                self.gpu_func[dim_grid_1d, dim_block_1d](
                                    field, species.x, species.y, species.z,
                                    t, amplitude, self.length_scale )
