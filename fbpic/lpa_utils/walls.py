# Copyright 2020, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Alberto de la Ossa
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines an abstract Wall class and the derived
classes, such as mirror, perfect electric conductor (PEC),
and perfect magnetic conductor (PMC) class.
"""
from abc import ABC
from scipy.constants import c

import math as m
import numpy as np

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed

if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_2d, compile_cupy


@cuda.jit(device=True, inline=True)
def grid_data_bilinear_interp(field, z, r, dz, dr, Nz, Nr, zmin):
    zi_min = int(m.floor((z - zmin) / dz))
    ri_min = int(m.floor(r / dr))

    if zi_min >= Nz - 1:
        zi_min = Nz - 2
    elif zi_min < 0:
        zi_min = 0
    if ri_min >= Nr - 1:
        ri_min = Nr - 2

    r1 = dr * ri_min
    z1 = zmin + dz * zi_min
    r2 = r1 + dr
    z2 = z1 + dz
    det = 1.0 / ((z2 - z1) * (r2 - r1))

    fQ11 = field[zi_min, ri_min]
    fQ12 = field[zi_min, ri_min + 1]
    fQ22 = field[zi_min + 1, ri_min + 1]
    fQ21 = field[zi_min + 1, ri_min]

    b1 = fQ11 * (r2 - r) + fQ12 * (r - r1)
    b2 = fQ21 * (r2 - r) + fQ22 * (r - r1)

    return det * ((z2 - z) * b1 + (z - z1) * b2)


@cuda.jit(device=True, inline=True)
def P00(x, h, L):
    return -(x - h) * (x + L) ** 2 / (h * L**2)


@cuda.jit(device=True, inline=True)
def P01(x, h, L):
    return x * (x + L) ** 2 / (h * (h + L) ** 2)


def P00n(x, h, L):
    return -(x - h) * (x + L) ** 2 / (h * L**2)


def P01n(x, h, L):
    return x * (x + L) ** 2 / (h * (h + L) ** 2)


@cuda.jit(device=True, inline=True)
def ray_casting(z, r, polygon):
    n = len(polygon)
    count = 0
    for i in range(n - 1):
        z1 = polygon[i][0]
        z2 = polygon[i + 1][0]
        r1 = polygon[i][1]
        r2 = polygon[i + 1][1]

        if (r < r1) != (r < r2) and z < (z2 - z1) * (r - r1) / (r2 - r1) + z1:
            count += 1

    return False if count % 2 == 0 else True


class Wall(ABC):
    def save_fields():
        pass

    def create_wall():
        pass

    def set_boundary_conditions():
        pass


class Mirror(Wall):
    def __init__(self, z_start, z_end, gamma_boost=None, m="all"):
        """
        Initialize a mirror.

        The mirror reflects the fields in the z direction, by setting the
        specified field modes to 0 in a thin slice orthogonal to z, at
        each timestep. By default, all modes are zeroed.

        Parameters
        ----------
        z_start: float
            Start position of the mirror in the lab frame

        z_end: float
            End position of the mirror in the lab frame

        gamma_boost: float
            For boosted-frame simulation: Lorentz factor of the boost

        m: int or list of ints
            Specify the field modes to set to zero
            By default, takes all modes to zero
        """

        self.z_start = z_start
        self.z_end = z_end
        self.gamma_boost = gamma_boost

        if m == "all":
            self.modes = None
        elif isinstance(m, int):
            self.modes = [m]
        elif isinstance(m, list):
            self.modes = m
        else:
            raise TypeError("m should be an int or a list of ints.")

    def save_fields(self, *args, **kwargs):
        pass

    def set_boundary_conditions(self, interp, comm, t_boost, *args, **kwargs):
        """
        Set the fields to 0 in a slice orthogonal to z

        Parameters:
        -----------
        interp: a list of InterpolationGrid objects
            Contains the values of the fields in interpolation space
        comm: a BoundaryCommunicator object
            Contains information on the position of the mesh
        t_boost: float
            Time in the boosted frame
        """
        # Lorentz transform
        if self.gamma_boost is None:
            z_start_boost, z_end_boost = self.z_start, self.z_end
        else:
            beta_boost = (1.0 - 1.0 / self.gamma_boost**2) ** 0.5
            z_start_boost = (
                1.0 /
                self.gamma_boost *
                self.z_start -
                beta_boost *
                c *
                t_boost)
            z_end_boost = 1.0 / self.gamma_boost * self.z_end \
                - beta_boost * c * t_boost

        # Calculate indices in z between which the field should be set to 0
        zmin, zmax = comm.get_zmin_zmax(
            local=True, with_guard=True, with_damp=True, rank=comm.rank
        )
        if (z_start_boost < zmin) or (z_start_boost >= zmax):
            return

        imax = int((z_start_boost - zmin) / interp[0].dz)
        n_cells = int((z_end_boost - z_start_boost) / interp[0].dz)
        imin = max(imax - n_cells, 0)

        # Set fields (E, B) to 0 on CPU or GPU
        for i, grid in enumerate(interp):
            if self.modes is not None:
                if i not in self.modes:
                    continue

            fieldlist = ["Er", "Et", "Ez", "Br", "Bt", "Bz"]
            if grid.use_pml:
                fieldlist = fieldlist + \
                    ["Er_pml", "Et_pml", "Br_pml", "Bt_pml"]
            for field in fieldlist:
                arr = getattr(grid, field)
                arr[imin:imax, :] = 0.0  # Uses numpy/cupy syntax


class PEC(Wall):
    def __init__(
        self,
        wall_arr,
        upper_segment,
        lower_segment,
        normal,
        tangent,
        side,
        z_start=None,
        h=None,
        L=None,
        dtv_filter=False,
        apply_pec=True,
    ):
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
        self.dtv_filter = dtv_filter
        self.apply_pec = apply_pec

    def create_wall(self, interp, comm, iteration):
        """
        Construct segments and penalty term for PEC boundary condition

        Parameters:
        -----------
        interp: a list of InterpolationGrid objects
            Contains the values of the fields in interpolation space
        """

        # Calculate indices in z between which the field should be set to 0
        zmin_global, zmax_global = comm.get_zmin_zmax(
            local=True, with_guard=True, with_damp=True, rank=comm.rank
        )

        z = cupy.linspace(zmin_global, zmax_global, interp[0].Nz)
        r = cupy.linspace(0.0, interp[0].rmax, interp[0].Nr)

        dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(interp[0].Nz, interp[0].Nr)
        self.segments = cupy.zeros(
            (interp[0].Nz, interp[0].Nr), dtype=np.int32)
        self.calculate_segments[dim_grid_2d, dim_block_2d](
            self.segments,
            z,
            r,
            self.wall_arr,
            self.upper_segment,
            self.lower_segment,
        )

    def set_boundary_conditions(
            self,
            interp,
            comm,
            iteration,
            *args,
            **kwargs):
        """
        Reflect fields at perfect electric conductor
        """
        if self.apply_pec:
            for m in range(len(interp)):
                r = cupy.linspace(interp[m].rmin, interp[m].rmax, interp[m].Nr)
                z = cupy.linspace(interp[m].zmin, interp[m].zmax, interp[m].Nz)

                dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(
                    interp[m].Nz, interp[m].Nr)

                self.pec_static_penalty_term[dim_grid_2d, dim_block_2d](
                    interp[m].Er,
                    interp[m].Et,
                    interp[m].Ez,
                    interp[m].Br,
                    interp[m].Bt,
                    interp[m].Bz,
                    r,
                    self.segments,
                    interp[m].rmax,
                )

                if self.dtv_filter and iteration % 100 == 0:
                    self.complexDTVFilter(
                        interp[m].Er, 9, 1, z, r, self.segments, self.normal)
                    self.complexDTVFilter(
                        interp[m].Et, 9, 1, z, r, self.segments, self.normal)
                    self.complexDTVFilter(
                        interp[m].Ez, 9, 1, z, r, self.segments, self.normal)

    @compile_cupy
    def calculate_local_coord_syst(
        s, z, r, wall_arr, upper_segment, lower_segment, normal, tangent, rmax
    ):
        i, j = cuda.grid(2)
        if i < s.shape[0] and j < s.shape[1]:
            if r[j] < rmax and r[j] > 0.0:
                n = len(wall_arr)
                for k in range(n - 1):
                    in_lower_segment = ray_casting(
                        z[i], r[j], lower_segment[k])
                    in_upper_segment = ray_casting(
                        z[i], r[j], upper_segment[k])
                    if in_lower_segment or in_upper_segment:
                        if (
                            lower_segment[k][0, 1] == rmax
                            and lower_segment[k][1, 1] == rmax
                        ):
                            break
                        if in_lower_segment:
                            z1 = lower_segment[k][0, 0]
                            r1 = lower_segment[k][0, 1]
                        else:
                            z1 = upper_segment[k][0, 0]
                            r1 = upper_segment[k][0, 1]

                        z2 = z[i]
                        r2 = r[j]

                        det_inv = 1.0 / (
                            normal[k, 0] * (-tangent[k, 1])
                            - (-tangent[k, 0]) * normal[k, 1]
                        )
                        if det_inv == 0:
                            break

                        Ainv_11 = det_inv * (-tangent[k, 1])
                        Ainv_12 = det_inv * (tangent[k, 0])

                        s[i, j] = Ainv_11 * (z2 - z1) + Ainv_12 * (r2 - r1)
                        break

    @compile_cupy
    def calculate_segments(
            segments,
            z,
            r,
            wall_arr,
            upper_segment,
            lower_segment):
        i, j = cuda.grid(2)
        if i < segments.shape[0] and j < segments.shape[1]:
            n = len(wall_arr)
            for k in range(n - 1):
                in_lower_segment = ray_casting(z[i], r[j], lower_segment[k])
                in_upper_segment = ray_casting(z[i], r[j], upper_segment[k])
                in_poly = ray_casting(z[i], r[j], wall_arr)
                if in_lower_segment:
                    segments[i, j] = 1
                    break
                elif in_upper_segment:
                    segments[i, j] = 2
                    break
                elif not in_poly:
                    segments[i, j] = 3
                    break

    @compile_cupy
    def pec_static_penalty_term(Er, Et, Ez, Br, Bt, Bz, r, segments, rmax):
        i, j = cuda.grid(2)
        if i < Ez.shape[0] and j < Ez.shape[1]:
            if r[j] < rmax:
                # segments: 2 upper, segments: 3 outside polygon
                if segments[i, j] == 2 or segments[i, j] == 3:
                    Er[i, j] = 0.
                    Et[i, j] = 0.
                    Ez[i, j] = 0.
                    Br[i, j] = 0.
                    Bt[i, j] = 0.
                    Bz[i, j] = 0.

    @compile_cupy
    def DTVStep(u0, U, v, a, dt, Lambda, z, r, segments, normal):
        i, j = cuda.grid(2)
        N = u0.shape[0]
        M = u0.shape[1]
        if i > 0 and j > 0 and i < N - 1 and j < M - 1:
            # if i < N-1 and j < M-1:
            u1 = U[i, j]
            u1_inv = 1 / u1
            if segments[i, j] == 1 or segments[i, j] == 2:
                Lambda = 15
                tol = 1.0e-3
            else:
                Lambda = 9
                tol = 1.0e-3

            if (
                abs((u1 - U[i + 1, j]) * u1_inv) > tol
                or abs((u1 - U[i - 1, j]) * u1_inv) > tol
                or abs((u1 - U[i, j + 1]) * u1_inv) > tol
                or abs((u1 - U[i, j - 1]) * u1_inv) > tol
            ):
                s = (
                    (U[i, j - 1] - u1) ** 2
                    + (U[i, j + 1] - u1) ** 2
                    + (U[i - 1, j] - u1) ** 2
                    + (U[i + 1, j] - u1) ** 2
                    + a
                ) ** 0.5

                sim = (
                    (U[i - 1, j - 1] - U[i - 1, j]) ** 2
                    + (U[i - 1, j + 1] - U[i - 1, j]) ** 2
                    + (U[i - 2, j] - U[i - 1, j]) ** 2
                    + (u1 - U[i - 1, j]) ** 2
                    + a
                ) ** 0.5

                sip = (
                    (U[i + 1, j - 1] - U[i + 1, j]) ** 2
                    + (U[i + 1, j + 1] - U[i + 1, j]) ** 2
                    + (u1 - U[i + 1, j]) ** 2
                    + (U[i + 2, j] - U[i + 1, j]) ** 2
                    + a
                ) ** 0.5

                sjm = (
                    (U[i, j - 2] - U[i, j - 1]) ** 2
                    + (u1 - U[i, j - 1]) ** 2
                    + (U[i - 1, j - 1] - U[i, j - 1]) ** 2
                    + (U[i + 1, j - 1] - U[i, j - 1]) ** 2
                    + a
                ) ** 0.5

                sjp = (
                    (u1 - U[i, j + 1]) ** 2
                    + (U[i, j + 2] - U[i, j + 1]) ** 2
                    + (U[i - 1, j + 1] - U[i, j + 1]) ** 2
                    + (U[i + 1, j + 1] - U[i, j + 1]) ** 2
                    + a
                ) ** 0.5

                v[i, j] = u1 + dt * (
                    (U[i + 1, j] - u1) * (1.0 + s / sip)
                    + (U[i - 1, j] - u1) * (1.0 + s / sim)
                    + (U[i, j + 1] - u1) * (1.0 + s / sjp)
                    + (U[i, j - 1] - u1) * (1.0 + s / sjm)
                    - Lambda * s * (u1 - u0[i, j])
                )

    def complexDTVFilter(self, u0, Lambda, timeSteps, z, r, segments, normal):
        a = 1e-6
        a = a**2
        N = u0.shape[0]
        M = u0.shape[1]
        dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(N, M)

        dt = 0.0002

        v = u0.real
        U = u0.real
        k = 1
        while k <= timeSteps:
            self.DTVStep[dim_grid_2d, dim_block_2d](
                u0.real, U, v, a, dt, Lambda, z, r, segments, normal
            )

            k = k + 1
            U = v
        u0.real = U

        v = u0.imag
        U = u0.imag
        k = 1
        while k <= timeSteps:
            self.DTVStep[dim_grid_2d, dim_block_2d](
                u0.imag, U, v, a, dt, Lambda, z, r, segments, normal
            )

            k = k + 1
            U = v
        u0.imag = U


class PMC(Wall):
    def __init__(
        self, z_start=None, h=None, L=None, gamma_boost=None, m="all", eta=0.25
    ):
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
        if m == "all":
            self.modes = None
        elif isinstance(m, int):
            self.modes = [m]
        elif isinstance(m, list):
            self.modes = m
        else:
            raise TypeError("m should be an int or a list of ints.")

    def save_fields(self, interp, *args, **kwargs):
        for i, grid in enumerate(interp):
            if self.modes is not None:
                if i not in self.modes:
                    continue

            self.Br_o = getattr(grid, "Br")
            self.Bt_o = getattr(grid, "Bt")
            self.Bz_o = getattr(grid, "Bz")

    def set_boundary_conditions(self, interp, comm, *args, **kwargs):
        """
        Reflect B field at perfect magnetic conductor
        """
        # Calculate indices in z between which the field should be set to 0
        zmin, zmax = comm.get_zmin_zmax(
            local=False, with_guard=True, with_damp=True, rank=comm.rank
        )

        z = cupy.linspace(zmin, zmax, interp[0].Nz)
        r = cupy.linspace(0.0, interp[0].rmax, interp[0].Nr)

        im = int((self.z_start - zmin) / interp[0].dz)
        imax = int((self.z_start + self.h - zmin) / interp[0].dz)
        imin = int((self.z_start - self.L - zmin) / interp[0].dz)

        R, Z = cupy.meshgrid(r, z)

        for i, grid in enumerate(interp):
            if self.modes is not None:
                if i not in self.modes:
                    continue

            Br = getattr(grid, "Br")
            Bt = getattr(grid, "Bt")
            Bz = getattr(grid, "Bz")

            Br[imin: im + 3, :] -= self.eta * (
                self.Br_o[imin: im + 3, :]
                - self.Br_o[imax, :]
                * (
                    P01(Z[imin: im + 3, :], self.h, self.L)
                )
            )

            Bt[imin: im + 3, :] -= self.eta * (
                self.Bt_o[imin: im + 3, :]
                - self.Bt_o[imax, :]
                * (
                    P01(Z[imin: im + 3, :], self.h, self.L)
                )
            )
            Bz[imin: im + 3, :] -= self.eta * (
                self.Bz_o[imin: im + 3, :]
                - self.Bz_o[im, :]
                * (
                    P00(Z[imin: im + 3, :], self.h, self.L)
                )
                - self.Bz_o[imax, :]
                * (
                    P01(Z[imin: im + 3, :], self.h, self.L)
                )
            )

            Br[imin: im + 3, -1] = 0.0
            Bt[imin: im + 3, -1] = 0.0
            Bz[imin: im + 3, -1] = 0.0
            Br[imin: im + 3, 0] = 0.0
            Bt[imin: im + 3, 0] = 0.0
            Br[im, :] = 0.0
            Bt[im, :] = 0.0
