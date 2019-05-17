#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Scheiber et al. [1]_ found the bond orientational order parameters of Steinhardt et al [2]_ to be useful looking at the
segregationof Re in W. This module calculates them from atomic positions.

.. [1] Scheiber, Razumovskiy, Puschnig, Pippan, and Romaner, Acta Mat 88 (2015)
.. [2] Steinhardt, Nelson, and Ronchetti, PRB 28 (1983)

Maybe someday it would be nice to have a fermi-smeared bond order, which took a weighted average based on a fermi
function of the distance instead of a straight mean across neighbours inside a hard cut. But that's not for today.
"""

import numpy as np
from scipy.special import sph_harm

__author__ = "Liam Huber, Yuji Ikeda"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def bond_orientational_order_parameter(pos, target_ids, xl_structure, latt, lmax=8, rcut=None):
    """
    Given a set of n atomic positions, calculates the bond-order parameters for each site in `target_ids` using a cut-
    off distance halfway between the 2nd and 3rd nearest neighbour distance.

    .. warning::

        Ignores periodic boundary conditions. This is fine for our CGB cluster calculations, but be careful if
        you want to use this code elsewhere.

    Args:
        pos (np.ndarray): :math:`(n, 3)` vector of atomic positions.
        target_ids (np.ndarray): :math:`(n',)` vector of ids in `pos` for which to calculate bond orientational order
                                 parameters.
        xl_structure (str): The crystal structure. Currently just `fcc` or `bcc`.
        latt (float): Lattice vector for the crystal.
        lmax (int): The maximum bond-order to go to (*l* in :math:`Y_l^m` for the spherical harmonics.)
        rcut (float): The cutoff for neighbours to be included in the bond counting (default is 3rd nearest neighbour).

    Returns:
        (*np.ndarray*) -- The bond-order orientation parameters with shape :math:`(n', l_{max})`.
    """

    # Bond-order parameters sum over all neighbours in a cutoff radius: default is 3NN
    if rcut is None:
        if xl_structure.lower() == 'fcc':
            rcut = np.sqrt(3./2.) * latt
        elif xl_structure.lower() == 'bcc':
            rcut = np.sqrt(2) * latt
        else:
            print ("Only FCC and BCC lattices are implemented at the moment.")
            raise NotImplementedError

    # For now let's do it with a nested list and see how slow that is...
    qs = np.empty((len(target_ids), lmax))
    for i in target_ids:
        # Calculate displacement vectors to neighbours of each target id
        disp = pos - pos[i]
        dist_mask = np.sum(disp * disp, axis=1) < rcut * rcut
        dist_mask[i] = False  # No self counting
        neigh_disp = disp[dist_mask]
        # Transform these displacements to spherical coordinates so they can be used in spherical harmonics
        _, theta, phi = to_spherical(neigh_disp[:, 0], neigh_disp[:, 1], neigh_disp[:, 2])
        # Calculate component spherical harmonics

        ql = []
        for l in np.arange(1, lmax + 1):
            # Make a matrix of spherical harmonics, with neighbours down rows, and m values along columns
            ylm_matrix = np.empty((len(neigh_disp), 2 * l + 1), dtype=np.complex128)
            for j, m in enumerate(np.arange(-l, l + 1)):
                ylm_matrix[:, j] = sph_harm(m, l, theta, phi)  # This is a 1d vector, but with real and imaginary parts
            ylm_mean = np.mean(ylm_matrix, axis=0)  # Averaging the Ylm over all neighbours
            ylm_mag_sq = np.real(ylm_mean * np.conj(ylm_mean))
            # Finally, we have all the pieces for the bond order parameter
            ql += [np.sqrt((4 * np.pi / (2 * l + 1.)) * np.sum(ylm_mag_sq))]
        qs[i, :] = np.array(ql)

    return qs


def to_spherical(x, y, z):
    """
    Converts Cartesian coordinates to spherical coordinates. Theta and phi are azimuthal and polar angles, respectively
    (as in SciPy). Pulled from StackOverflow_.

    Args:
        x (float): Cartesian x-coordinate.
        y (float): Cartesian y-coordinate.
        z (float): Cartesian z-coordinate.
    Returns:
        3-element tuple containing spherical coordinates

        - (*float*) -- Radius.
        - (*float*) -- Azimuthal angle.
        - (*float*) -- Polar angle.

    .. _StackOverflow: https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    r_circ_sq = x * x + y * y
    r_sphere = np.sqrt(r_circ_sq + z * z)
    theta = np.arctan2(y, x)  # Azimuthal, in [0, 2*pi]
    # Polar from the noth pole, in [0, pi]
    phi = 0.5 * np.pi - np.arctan2(z, np.sqrt(r_circ_sq))
    return r_sphere, theta, phi
