#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Generates a cluster of atoms, presumably with a grain boundary running down the middle (x-axis normal), but bulk-like
configurations are also possible.

A couple FCC CSL parameters for quick reference:

- 53.1301024<001>(210) Sigma 5 tilt
- 36.8698976<001>(310) Sigma 5 tilt

This page_ is also useful.

.. _page: https://www.researchgate.net/figure/257823895_fig3_FIG-4-110-symmetric-tilt-grain-boundary-structures-with-structural-units-outlined-for

.. todo::

    Right now everything assumes a cubic lattice. I think we need to do lots of multiplying by the bravais vectors to
    get it to work more generally.
"""

import numpy as np
from scipy.spatial.distance import cdist
from . import extra_math as em
from .osio import tee

__author__ = "Liam Huber, Raheleh Hadian"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"

N_MEM_SWITCH = 234000000  # Largest array size allowed when looking for pairs to merge before switching algorithms
# The default value of 234000000 should limit memory usage to somewhere around 2GB


def find_bravais(misorientation, shared, normal, symmetric, bravais, precision, verbosity, max_steps):
    """
    Get the bravais lattices for two grains with the provided macroscopic parameters. Bravais lattices will be rotated
    together until the GB plane is the in yz-plane--i.e. the macroscopic parameters will be as requested but perhaps
    with a coordinate change applied to both grains.

    Args:
        misorientation float: Misorientation between the two grains.
        shared (np.ndarray): :math:`(3,)` axis about which to apply misorientation.
        normal (np.ndarray): :math:`(3,)` grain boundary plane normal for one of the grains.
        symmetric (bool): Whether to ignore any normal vector and simply make a symmetric tilt boundary.
        bravais (np.ndarray) : :math:`(3,3)` matrix of lattice vectors for the underlying unit cell, one in each row.
        precision (float): Numeric precision, e.g. for determining if two vectors are close enough (L2) to be the same.
        verbosity (bool): Whether to print info from called subroutines (e.g. searching for rotation matrices).
        max_steps (int): Maximum allowable steps to use in numeric search subroutines.

    Returns:
        3-element tuple containing

        - (*np.ndarray*) -- :math:`(3,3)` Bravais vectors for the left grain (x<0).
        - (*np.ndarray*) -- :math:`(3,3)` Bravais vectors for the right grain (x>0).
        - (*np.ndarray*) -- :math:`(3,)` Shared axis in the transformed coordinates.
    """
    # Get new bravais vectors for the two rotated grains
    if not symmetric:
        brav1, brav2, new_shared = _calculate_bravais(misorientation, shared, normal, bravais, precision, verbosity,
                                                      max_steps)
    else:
        brav1, brav2, new_shared = _bravais_from_symmetry(misorientation, shared, bravais)

    return brav1, brav2, new_shared


def ideal_hemispheres(brav1, brav2, basis, translation, radius):
    """
    Generates two arrays of atoms, one for each grain. The grain boundary (GB) normal is (100) and both grains
    together form a sphere.

    First, two appropriately rotated bravais lattices are found, one for each grain. Unless the `symmetric` flag is
    thrown, the first grain will have the `normal` vector pointing in the x-direction. Next, a large cubic repeat is
    made of these bravais lattices and populated with atoms using the bases. For each grain, these atoms are then
    trimmed to each form a hemisphere.

    .. warning::

        The hemispheres are allowed to extend slightly into each other and some atoms from each grain may overlap.

    .. warning::

        Right now it all only works with cubic lattices.

    Args:
        brav1 (np.ndarray): :math:`(3,3)` lattice vectors for the unit cell of one grain, one in each row.
        brav2 (np.ndarray): :math:`(3,3)` lattice vectors for the unit cell of the other grain, one in each row.
        basis (np.ndarray): :math:`(n,3)` direct-coordinate location of atoms in the unit cell.
        translation (np.ndarray): :math:`(3,)` relative translation of one grain to the other (in same units as the
                                  Bravais).
        radius (float): Radius of the cluster of atoms desired.

    Returns:
        2-element tuple of Cartesian atomic positions for the sphere centred at the origin,

        - (*np.ndarray*): :math:`(n,3)` Cartesian coordinates of left hemisphere.
        - (*np.ndarray*): :math:`(n,3)` Cartesian coordinates of right hemisphere.
    """

    # Find shortest bravais vector
    shortest = em.shortest_vector(brav1)  # Because we look for a magnitude, it doesn't matter if we use brav1 or brav2

    # Assuming a cubic bravais lattice, the rotated bravais make some sort of arbitrarily oriented rectangular prism
    # We now want to make sure we have a rectangular prism which encloses our sphere
    # Without being too small, we can suppose the rectangular prism is a cube of the shortest dimensions
    # The largest sphere that can be inscribed is one whose radius is half the side length
    # So a cube with 2 * radius in side length is good enough

    reps = 2 * radius / shortest
    reps += 1  # Safety first

    # Make a list of these lattice vector multipliers
    rep_range = np.arange(-int(reps / 2.), int(reps / 2.) + 1)
    xx, yy, zz = np.meshgrid(rep_range, rep_range, rep_range)
    lattice_mults = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).transpose()

    # For each grain, construct a hemisphere of atoms (centered about (0,0,0) instead of the box center)
    bravs = [brav1, brav2]
    hemispheres = []
    r_sq = radius ** 2

    # What do I want? To shift one infinite grain relative to the other, then still cut in the same place
    lattice_space_shift = translation * np.matrix(np.linalg.inv(bravs[1]))
    lattice_space_shift -= lattice_space_shift.astype(int)  # Get minimum periodic tranlsation
    lattice_shifts = [np.zeros(3), lattice_space_shift]  # This is in lattice space though...
    #print "real-space lattice shift requested and obtained", translation, lattice_shifts[1] * np.matrix(bravs[1])

    for i in range(2):
        brav = bravs[i]
        lattice_shift = lattice_shifts[i]

        unshifted_atom_pos = np.array((lattice_mults + lattice_shift) * np.matrix(brav))

        atom_pos = np.empty((0, 3))
        for basis_vec in basis:
            shift = basis_vec * brav
            atom_pos = np.append(atom_pos, unshifted_atom_pos + shift, axis=0)
            atom_pos = np.array(atom_pos)  # Array-type for element-wise multiplication

        dist_sq_from_center = np.sum(atom_pos * atom_pos, axis=1)
        radius_mask = dist_sq_from_center < r_sq
        atom_pos = atom_pos[radius_mask]  # An unrotated sphere of atoms centered at 0, 0, 0 with the correct radius

        if i == 0:
            mask_hemisphere = atom_pos[:, 0] < 0.01 * shortest
        else:
            mask_hemisphere = atom_pos[:, 0] > -0.01 * shortest
        hemispheres += [atom_pos[mask_hemisphere]]

    pos1 = hemispheres[0]
    pos2 = hemispheres[1]

    return pos1, pos2


def merge_hemispheres(pos1, pos2, merge_dist):
    """
    Given two hemispheres of atoms, return a single array in which atoms from the two hemispheres which are too close
    too each other have been deleted and replaced with a new atom at their average position (i.e. exactly on the
    boundary plane.)

    Args:
        pos1 (np.ndarray): :math:`(n,3)` Cartesian position of atoms in one hemisphere.
        pos2 (np.ndarray): :math:`(m,3)` Cartesian position of atoms in the other hemisphere.
        merge_dist (float): Distance within which to merge atoms in opposite hemispheres.

    Returns:
        (*np.ndarray*) -- :math:`(\\leq m+n, 3)` Merged atoms.
    """

    # Merge atoms from different grains that are too close
    safety_scale = 1.1
    near_interface_mask1 = pos1[:, 0] > -merge_dist * safety_scale
    near_interface_mask2 = pos2[:, 0] < merge_dist * safety_scale

    subset1 = pos1[near_interface_mask1]
    subset2 = pos2[near_interface_mask2]

    subset_ids1 = np.arange(len(pos1))[near_interface_mask1]
    subset_ids2 = np.arange(len(pos2))[near_interface_mask2]

    # We might need to trade time for memory if the number of possible merge pairs to check is too big
    # TODO: Introduce binning to avoid this problem? Is it even a problem for most applications?
    n_subset1 = len(subset1)
    n_subset2 = len(subset2)
    n_pairs = n_subset1 * n_subset2
    if n_pairs == 0:
        return np.append(pos1, pos2, axis=0)

    # If there are too many pairs, look for neighbours with a loop
    if n_pairs > N_MEM_SWITCH:

        closest_ids1 = np.arange(n_subset1)
        closest_ids2 = np.empty(n_subset1, dtype=int)
        closest_dist = np.empty(n_subset1)
        for i, loc in enumerate(subset1):
            diff = subset2 - loc
            j = np.sum(diff * diff, axis=1).argmin()
            dist = np.sqrt(np.sum(diff[j] * diff[j]))

            closest_ids2[i] = int(j)
            closest_dist[i] = dist
    # But if there are few enough pairs, we can use scipy to get directly a full array of the distances
    else:
        pair_distances = cdist(subset1, subset2, metric='euclidean')
        closest_ids1 = np.arange(len(pair_distances))
        try:
            closest_ids2 = pair_distances.argmin(axis=1)
        except ValueError:
            print("Pair distances:\n", pair_distances)
            print("n_subset1, subset1:\n", n_subset1, subset1)
            print("n_subset2, subset2:\n", n_subset2, subset2)
            print("n_pairs", n_pairs)
            raise
        closest_dist = pair_distances[closest_ids1, closest_ids2]

    merge_mask = closest_dist <= merge_dist
    merge_ids1 = subset_ids1[closest_ids1[merge_mask]]
    merge_ids2 = subset_ids2[closest_ids2[merge_mask]]

    # Create new merged atoms
    merged = 0.5 * (pos1[merge_ids1] + pos2[merge_ids2])
    # And delete the parents of these atoms
    pos1 = np.delete(pos1, merge_ids1, axis=0)
    pos2 = np.delete(pos2, merge_ids2, axis=0)

    pos = np.append(np.append(merged, pos1, axis=0), pos2, axis=0)

    return pos


def _bravais_from_symmetry(misorientation, shared, bravais):
    """
    Calculate two bravais lattice which will form a symmetric tilt boundary given the misorientation and shared axis.

    Args:
        misorientation (float): Misorientation between the two grains.
        shared (np.ndarray): :math:`(3,)` axis about which to apply misorientation.
        bravais (np.ndarray): :math:`(3,3)` lattice vectors for the unit cell, one in each row.

    Returns:
        3-element tuple containing

        - (*np.ndarray*) -- The rotated Bravais lattice for one grain.
        - (*np.ndarray*) -- And for the other.
        - (*np.ndarray*) -- The shared axis using the same coordinates as the returned Bravais lattices.
    """
    shared = em.l2normalize(shared)

    misorientation_mat1 = em.rotation(0.5 * misorientation, shared)
    misorientation_mat2 = em.rotation(-0.5 * misorientation, shared)

    x_axis = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])
    # If the shared axis already lies in the grain boundary (GB) plane (x-axis), no alignment is necessary
    if em.vectors_are_perpendicular(shared, x_axis):
        alignment_mat = np.identity(3)
    # If the shared axis is perfectly parallel to the GB plane, we have a <001> tilt boundary
    # Simply rotate the shared axis to the z-axis
    elif em.vectors_are_parallel(shared, x_axis):
        alignment_mat, _, _ = em.alignment_matrix(shared, z_axis)
    # Otherwise we're somewhere in between and we want to make sure the shared axis winds up somewhere in the GB plane
    else:
        perp_to_shared = em.l2normalize(np.cross(x_axis, shared))
        alignment_mat, _, _ = em.alignment_matrix(perp_to_shared, x_axis)

    rot_mat1 = em.matmat_mult(alignment_mat, misorientation_mat1)
    rot_mat2 = em.matmat_mult(alignment_mat, misorientation_mat2)

    brav1 = em.matvec_mult(rot_mat1, bravais)
    brav2 = em.matvec_mult(rot_mat2, bravais)

    return brav1, brav2, shared


def _calculate_bravais(misorientation, shared, normal, bravais, precision, verbosity, max_steps):
    """
    Get the two bravais lattices for grains having a boundary defined by the macroscopic GB parameters `misorientation`
    <`shared`> (`normal`).

    .. todo::

        Are lattice vectors rows or columns of bravais? Only tested for cubic Bravais...

    .. todo::

        There's some code repeat here between this and _bravais_from_symmetry that could be refactored out.

    Args:
        misorientation (float): Relative misorientation (in rads) of two grains about shared axis.
        shared (np.ndarray): :math:`(3,)` axis about which to apply misorientation.
        normal (np.ndarray): :math:`(3,)` grain boundary plane normal for one of the grains.
        bravais (np.ndarray): :math:`(3,3)` lattice vectors for the unit cell, one in each row.
        precision (float): Numeric precision, e.g. for determining if two vectors are close enough (L2) to be the same.
        verbosity (bool): Whether to print info from Newton's method (if any).
        max_steps (int): Maximum allowable steps to use in Newton's method.

    Returns:
        3-element tuple containing

        - (*np.ndarray*) -- :math:`(3,3)` Bravais lattice for the first grain.
        - (*np.ndarray*) -- :math:`(3,3)` and the second grain.
        - (*np.ndarray*) -- :math:`(3,)` the shared axis in the new coordinates.
    """

    x_axis = np.array([1, 0, 0])
    shared = em.l2normalize(shared)
    normal = em.l2normalize(normal)

    misorientation_mat1 = em.rotation(0.5 * misorientation, shared)
    misorientation_mat2 = em.rotation(-0.5 * misorientation, shared)

    new_normal = em.matvec_mult(misorientation_mat1, normal)
    perp_rot, _, _ = em.alignment_matrix(new_normal, x_axis,
                                         precision=precision, verbosity=verbosity, max_steps=max_steps)

    rot_mat1 = em.matmat_mult(perp_rot, misorientation_mat1)
    rot_mat2 = em.matmat_mult(perp_rot, misorientation_mat2)
    new_shared = em.matvec_mult(rot_mat1, shared)

    brav1 = em.matvec_mult(rot_mat1, bravais)
    brav2 = em.matvec_mult(rot_mat2, bravais)

    return brav1, brav2, new_shared
