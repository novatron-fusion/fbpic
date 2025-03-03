# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in Compton scattering (on GPU).
"""
import random
import math as m
from scipy.constants import c, k, h, epsilon_0, e
from fbpic.utils.threading import njit_parallel, prange


@njit_parallel
def pairs_per_cell_numba(N_batch, array_in1, array_in2, array_out, intra):
	"""
	Calculate the number of pairs per cell
	"""
	for i in prange(N_batch):
		if array_in1[i] == 0 or array_in2[i] == 0:
			array_out[i] = 0
		else:
			if intra:
				if array_in1[i] < 2: 
					array_out[i] = 0
				else:
					array_out[i] = array_in1[i] / 2 \
						if array_in1[i] % 2 else (array_in1[i] + 1) / 2
			else:
				if array_in1[i] > array_in2[i]:
					array_out[i] = array_in1[i]
				else:
					array_out[i] = array_in2[i]


@njit_parallel
def density_per_cell_numba(N_batch, density, weights, npart,
                          prefix_sum, d_invvol, Nz):
    """
    Calculate density of species per cell
    """
    for i in prange(N_batch):
        sum = 0.
        if i > 0:
            i_min = int(prefix_sum[i-1])
        else:
            i_min = 0
        for j in range(int(npart[i])):
            sum += weights[i_min + j]
        invvol = d_invvol[int(i / Nz)]
        density[i] = sum * invvol


@njit_parallel
def n12_per_cell_numba(N_batch, n12, w1, w2,
                      npairs, shuffled_idx1, shuffled_idx2,
                      prefix_sum_pair, prefix_sum1, prefix_sum2,
                      d_invvol, Nz):
    """
    Calculate n12 of species per cell
    n12 is the sum of minimum species weights
    """
    # Loop over cells
    # Loop over batch of particle pairs and perform collision
    for i in prange(N_batch):
        # Loop through the batch
        if i > 0:
            i_min1 = int(prefix_sum1[i-1])
            i_min2 = int(prefix_sum2[i-1])
            p_min = int(prefix_sum_pair[i-1])
        else:
            i_min1 = 0
            i_min2 = 0
            p_min = 0
        sum = 0.
        for j in range(int(npairs[i])):
            si1 = shuffled_idx1[p_min + j]
            si2 = shuffled_idx2[p_min + j]
            if w1[i_min1+si1] < w2[i_min2+si2]:
                sum += w1[i_min1+si1]
            else:
                sum += w2[i_min2+si2]
        invvol = d_invvol[int(i / Nz)]
        n12[i] = sum * invvol


@njit_parallel
def temperature_per_cell_numba(N_batch, T, npart,
                              prefix_sum, mass,
                              ux, uy, uz):
    """
    Calculate temperature of species in cell
    """
    for i in prange(N_batch):
        if npart[i] == 0:
            T[i] = 0.
        else:
            if i > 0:
                i_min = int(prefix_sum[i-1])
            else:
                i_min = 0

            vx_mean = 0.
            vy_mean = 0.
            vz_mean = 0.
            v2 = 0.
            for j in range(int(npart[i])):
                vx_mean += ux[i_min + j] * c
                vy_mean += uy[i_min + j] * c
                vz_mean += uz[i_min + j] * c

                v2 += (ux[i_min + j]**2 + uy[i_min + j]
                       ** 2 + uz[i_min + j]**2) * c**2

            invNp = (1. / npart[i])
            v2 *= invNp
            vx_mean *= invNp
            vy_mean *= invNp
            vz_mean *= invNp
            u_mean2 = vx_mean**2 + vy_mean**2 + vz_mean**2
            udiff = (v2 - u_mean2)
            if udiff < 0.:
                T[i] = 0.
            else:
                T[i] = (mass / (3. * k)) * udiff


@njit_parallel
def get_cell_idx_per_pair_numba(N_batch, cell_idx, npair, prefix_sum_pair):
    """
    Get the cell index of each pair.
    The cell index is 1d and calculated by:
    prefix sum of pairs

    Parameters
    ----------
    cell_idx : 1darray of integers
                The cell index of the pair

    npair : 1darray of integers
                The particle pair array

    prefix_sum_pair : 1darray of integers
                Cumulative prefix sum of
                the particle pair array		
    """
    for i in prange(N_batch):
        if i > 0:
            s = prefix_sum_pair[i-1]
        else:
            s = 0
        p = npair[i]
        k = 0
        while k < p:
            cell_idx[s+k] = i
            k += 1


@njit_parallel
def get_shuffled_idx_per_particle_numba(N_batch, shuffled_idx, npart,
                            			npair, prefix_sum_pair,
										intra, species):
    """
    Get the shuffled index of each particle in the array of pairs.
    The particles are shuffled with a linear congruential generator
    that is cyclic and non-repeating.

    Parameters
    ----------
    shuffled_idx : 1darray of integers
            The shuffled particle index of the particle

    npart : 1darray of integers
            The particle array

    npair : 1darray of integers
            The particle pair array

    prefix_sum_pair : 1darray of integers
            Cumulative prefix sum of
            the pair array	

    random_states : random states of the random generator

    intra : boolean
            True: like-particle collisions
            False: unlike-particle collisions

    species : int
            species number: 1 or 2
    """
    for i in prange(N_batch):
        if i > 0:
            s = prefix_sum_pair[i-1]
        else:
            s = 0
        
        start = 0
        stop = npart[i]
        if intra and species == 1:
            stop = npair[i]
        if intra and species == 2:
            start = npair[i]

        p = range(start, stop)
        
        random.shuffle(p)

        shuffled_idx[s+start:s+stop] = p


@njit_parallel
def perform_collisions_numba(N_batch, prefix_sum1, prefix_sum2,
                            shuffled_idx1, shuffled_idx2, cell_idx,
                            n1, n2, T1, T2,
                            n12, m1, m2,
                            q1, q2, w1, w2,
                            ux1, uy1, uz1,
                            ux2, uy2, uz2,
                            dt, coulomb_log,
                            period, debug,
                            param_s, param_logL):
    """
    Perform collisions between all pairs in each cell
    """
    # Loop over cells
    # Loop over batch of particle pairs and perform collision
    for ip in prange(N_batch):
        # The particles are randomly paired in each cell
        # with a linear congruential generator which
        # shuffles the particles in a non-repeating manner.
        # Note: plasma heating due to collisions can
        # be reduced in certain cases with a 'nearest neighbor'
        # (NN) pairing and/or low-pass filter. NN can be 
        # achieved with a finer grid in the particle sorting.
        cell = cell_idx[ip]
        if cell > 0:
            si1 = int(shuffled_idx1[ip] + prefix_sum1[cell-1])
            si2 = int(shuffled_idx2[ip] + prefix_sum2[cell-1])
        else:
            si1 = int(shuffled_idx1[ip])
            si2 = int(shuffled_idx2[ip])

        # Calculate Lorentz factor
        gamma1 = m.sqrt(1. + (ux1[si1]**2 + uy1[si1]**2 + uz1[si1]**2))
        gamma2 = m.sqrt(1. + (ux2[si2]**2 + uy2[si2]**2 + uz2[si2]**2))

        m12 = m1 / m2

        g12 = m12 * gamma1 + gamma2
        inv_g12 = 1. / g12

        # Center of mass (COM) velocity
        COM_vx = (m12 * ux1[si1] + ux2[si2]) * inv_g12
        COM_vy = (m12 * uy1[si1] + uy2[si2]) * inv_g12
        COM_vz = (m12 * uz1[si1] + uz2[si2]) * inv_g12

        COM_v_u1 = (COM_vx * ux1[si1] + COM_vy *
                    uy1[si1] + COM_vz * uz1[si1])
        COM_v_u2 = (COM_vx * ux2[si2] + COM_vy *
                    uy2[si2] + COM_vz * uz2[si2])

        COM_v2 = COM_vx**2 + COM_vy**2 + COM_vz**2
        COM_gamma = 1. / m.sqrt(1. - COM_v2)

        # momenta in COM
        term0 = (COM_gamma - 1.) * COM_v_u1 / COM_v2 - COM_gamma * gamma1
        ux_COM = ux1[si1] + COM_vx * term0
        uy_COM = uy1[si1] + COM_vy * term0
        uz_COM = uz1[si1] + COM_vz * term0

        u2_COM = ux_COM**2 + uy_COM**2 + uz_COM**2
        u_COM = m.sqrt(u2_COM)
        invu_COM2 = 1. / u2_COM

        # Lorentz transforms
        gamma1_COM = (gamma1 - COM_v_u1) * COM_gamma
        gamma2_COM = (gamma2 - COM_v_u2) * COM_gamma

        # Calculate coulomb log if not user-defined
        qqm = q1 * q2 / m1
        qqm2 = qqm**2
        logL = coulomb_log
        if logL <= 0.:
            coeff = 1. / (4. * m.pi * epsilon_0 * c**2)
            b0 = abs(coeff * qqm * COM_gamma * inv_g12 *
                        (gamma1_COM * gamma2_COM * invu_COM2 + m12))
            bmin = max(0.5 * h / (m1 * c * u_COM), b0)
            if T1[cell] == 0 or T1[cell] == 0:
                logL = 0.
            else:
                Debye2 = (epsilon_0 * k / e**2) / \
                    (n1[cell] / T1[cell] + n2[cell] / T2[cell])
                logL = 0.5 * m.log(1. + Debye2 / bmin**2)
                if logL < 2.:
                    logL = 2.

        coeff1 = 1. / (4. * m.pi * epsilon_0**2 * c**3)
        term1 = n1[cell] * n2[cell] * period * dt / n12[cell]
        term2 = coeff1 * qqm2 / (gamma1 * gamma2)
        term3 = COM_gamma * inv_g12 * u_COM
        term4 = (gamma1_COM * gamma2_COM * invu_COM2 + m12)

        # Calculate the collision parameter s12
        s12 = logL * term1 * term2 * term3 * term4 * term4

        # Low temperature correction
        v_rel = abs(g12 * u_COM * c / ( COM_gamma * gamma1_COM * gamma2_COM ))
        s_prime = (4.*m.pi/3)**(1/3) * term1 * \
            ((m1 + m2) / max(m1 * n1[cell]**(2/3), m2 * n2[cell]**(2/3))) * \
            v_rel
        if s12 < 0 or s_prime < 0:
            s = 0.
        else:
            s = min(s12, s_prime)

        if debug:
            param_s[ip] = s
            param_logL[ip] = logL

        # Random azimuthal angle
        phi = random.random()* 2.0 * m.pi

        # Calculate the deflection angle
        U = random.random()    # random float [0,1]
        if s12 < 4:
            a = 0.37 * s - 0.005 * s**2 - 0.0064 * s**3
            sin2X2 = a * U / m.sqrt(1 - U + a**2 * U)
            cosX = 1. - 2. * sin2X2
            sinX = 2. * m.sqrt(sin2X2 * (1. - sin2X2))
        else:
            cosX = 2. * U - 1.
            sinX = m.sqrt(1. - cosX * cosX)
        sinXcosPhi = sinX * m.cos(phi)
        sinXsinPhi = sinX * m.sin(phi)

        u_perp_COM = m.sqrt(ux_COM**2 + uy_COM**2)
        invu = 1. / u_perp_COM

        # Resulting momentum in COM frame
        if (u_perp_COM > 1.e-10 * u_COM):
            uxf1_COM = (ux_COM * uz_COM * invu * sinXcosPhi
                        - uy_COM * u_COM * invu * sinXsinPhi
                        + ux_COM * cosX)
            uyf1_COM = (uy_COM * uz_COM * invu * sinXcosPhi
                        + ux_COM * u_COM * invu * sinXsinPhi
                        + uy_COM * cosX)
            uzf1_COM = (-u_perp_COM * sinXcosPhi
                        + uz_COM * cosX)
        else:
            uxf1_COM = u_COM * sinXcosPhi
            uyf1_COM = u_COM * sinXsinPhi
            uzf1_COM = u_COM * cosX

        vC_ufCOM = (COM_vx * uxf1_COM
                    + COM_vy * uyf1_COM
                    + COM_vz * uzf1_COM)

        U1 = random.random()   # random float [0,1]
        if U1 * w1[si1] < w2[si2]:
            # Deflect particle 1
            term0 = (COM_gamma - 1.) * vC_ufCOM / COM_v2 + gamma1_COM * COM_gamma
            ux1[si1] = uxf1_COM + COM_vx * term0
            uy1[si1] = uyf1_COM + COM_vy * term0
            uz1[si1] = uzf1_COM + COM_vz * term0

        if U1 * w2[si2] < w1[si1]:
            # Deflect particle 2 (pf2 = -pf1)
            term0 = -(COM_gamma - 1.) * m12 * vC_ufCOM / COM_v2 + gamma2_COM * COM_gamma
            ux2[si2] = -uxf1_COM * m12 + COM_vx * term0
            uy2[si2] = -uyf1_COM * m12 + COM_vy * term0
            uz2[si2] = -uzf1_COM * m12 + COM_vz * term0