#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
For reading and writing LAMMPS-compatible structure files.
"""

import numpy as np
from . import osio

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def comment_count(counts):
    """
    Convert raw counts of atoms in the different cluster regions to a human-readable string.

    Args:
        counts (list): *int* values for the number of nonbulklike atoms, atoms in the inner sphere (excluding
                       non-bulklike), the shell between the inner and outer sphere, and the balance of atoms in the
                       cluster.

    Returns:
        (*str*) -- A comment describing the atom counts.
    """
    return 'atom counts for nonbulklike, inner, outer, balance = ' + ' '.join(str(c) for c in counts) + '; '


def comment_radii(r_inner, r_outer, r):
    """
    Convert radii defining the cluster and sub-spheres to a human-readable string.

    Args:
        r_inner (float): Inner sub-sphere radius.
        r_outer (float): Outer sub-sphere radius.
        r (float): Radius of the entire cluster.

    Returns:
        (*str*) -- A comment describing the cluster radii.
    """
    return 'inner, outer, and cluster radii = ' + ' '.join(str(rad) for rad in [r_inner, r_outer, r]) + '; '


def read_xyz(fin, step=-1):
    """
    Reads a LAMMPS-dumped .xyz file.

    Args:
        fin (str): File to read.
        step (int): Which frame to read. Can be negative to read from the end of the file. (Default is -1.)

    Returns:
        2-element tuple containing

        - (*np.ndarray*) -- :math:`(n, 3)` cartesian xyz positions of all atoms. and their species ids.
        - (*np.ndarray*) -- :math:`(n,)` and their *int* species flag.
    """

    nlines = osio.count_lines(fin)

    with open(fin, 'r') as f:
        natoms = int(float(f.readline()))

    lines_per_frame = natoms + 2.
    frames = nlines / lines_per_frame
    if frames != int(frames):
        raise Exception(fin + ' does not have a valid number of complete frames.')

    if (step > frames - 1) or (-step > frames):
        raise Exception('Trying to access a frame (' + str(step) + ') that doesn\'t exist (among '+ str(int(frames))
                        + ')')

    if step >= 0:
        nhead = step * lines_per_frame
        ntail = (frames - 1 - step) * lines_per_frame
    else:
        nhead = (frames + step) * lines_per_frame
        ntail = (-step - 1) * lines_per_frame

    data = np.genfromtxt(fin, skip_header=int(nhead) + 2, skip_footer=int(ntail))
    species = data[:, 0].astype(int)
    pos = data[:, 1:]

    return pos, species


def read_xyzin(fin):
    """
    Reads '.xyzin' file (the format for LAMMPS to read from when loading a structure.)

    Args:
        fin (str): File to read.

    Returns:
        5-element tuple containing

        - (*np.ndarray*) -- :math:`(n, 3)` cartesian xyz positions of all atoms. and their species ids.
        - (*np.ndarray*) -- :math:`(n,)` and their *int* species flag.
        - (*float*), (*float*), (*float*) -- The x-, y-, and z-lengths of the rectangular simulation cell.
    """

    with open(fin, 'r') as f:
        f.readline()  # Header
        f.readline()  # Blank
        f.readline()  # Number of atoms
        f.readline()  # Number of species
        f.readline()  # Blank
        x_line = f.readline()
        y_line = f.readline()
        z_line = f.readline()

    dx = _parse_xyzin_box_lines(x_line)
    dy = _parse_xyzin_box_lines(y_line)
    dz = _parse_xyzin_box_lines(z_line)

    data = np.genfromtxt(fin, skip_header=11)
    species = data[:, 1].astype(int)
    pos = data[:, 2:]

    return pos, species, dx, dy, dz


def reorder(pos, center, r_inner, r_outer, cna_vals=None, lattice_type=None):
    """
    Reorders a set of cartesian atomic positions positions so that non-bulk-like atoms are at the beginning of the
    position array, followed by inner atoms, followed by outer, followed by the balance of the list.

    - Non-bulk-like) Has a `cna_vals` id that doesn't correspond to the bulk-indicator for the given `lattice_type`. If
                    `cna_vals` is left as the default `None` type, this will be an empty list of positions.
    - Inner) The atoms whose position is within `r_inner` of the `center` point, but who are bulk-like.
    - Outer) Atoms who lie between `r_inner` and `r_outer` from `center`, whether they're bulk-like or not.
    - Balance) The balance of atoms in the pos list, all farther that `r_outer` from `center`.

    Args:
        pos (np.ndarray): :math:`(n,3)` Cartesian atom positions.
        center (np.ndarray): :math:`(3,)` center of the cluster.
        r_inner (float): Inner radius for partitioning the cluster.
        r_outer (float): Outer radius for partitioning the cluster.
        cna_vals (np.ndarray): :math:`(n,)` *int* LAMMPS common-neighbour analysis flags per atom. (Default has no
                               non-bulk-like atoms.)
        lattice_type (str): The crystal lattice type ('fcc' or 'bcc'.)

    Returns:
        2-element tuple containing

        - (*np.ndarray*) -- :math:`(n,3)` atomic positions with atoms ordered non-bulklike, inner, outer, and balance.
        - (*np.ndarray*) -- :math:`(4,)` *int* counts of atoms in the same order.
    """

    if cna_vals is not None:
        if lattice_type == 'fcc':
            bulk_code = 1
        elif lattice_type == 'bcc':
            bulk_code = 3
        elif lattice_type == 'hcp':
            bulk_code = 2
        else:
            raise ValueError('Lattice type not recognized.')
        cna_mask = cna_vals != bulk_code
    else:
        cna_mask = np.zeros(len(pos), dtype=bool)

    dist_sq = np.sum((pos - center)**2, axis=1)
    inner_mask = dist_sq <= r_inner**2
    outer_mask = (dist_sq <= r_outer**2) * (dist_sq > r_inner**2)
    balance_mask = np.logical_not(np.logical_or(inner_mask, outer_mask))

    nonbulklike = pos[np.logical_and(cna_mask, inner_mask)]
    inner = pos[np.logical_and(np.logical_not(cna_mask), inner_mask)]
    outer = pos[outer_mask]
    balance = pos[balance_mask]

    new_pos = np.vstack((nonbulklike, inner, outer, balance))
    if len(pos) != len(new_pos):
        raise Exception('pos was ' + str(len(pos)) + ' long, but new_pos has length ' + str(len(new_pos)))
    counts = np.array([len(nonbulklike), len(inner), len(outer), len(balance)], dtype=int)
    return new_pos, counts


def write_vorin(fout, pos):
    """
    Write a file with atomic position data formatted for voro++ to read.

    Args:
        fout (str): Name of file to write to.
        pos (np.ndarray): :math:`(n,3)` Cartesian atomic positions to write.
    """
    ids = np.arange(len(pos)).reshape(-1, 1)
    data = np.hstack((ids, pos))
    np.savetxt(fout, data, fmt='%d %f %f %f')


def write_xyzin(fout, pos, lx, ly=None, lz=None, species=None, nspecies=None,
                comment='Written with clustergb', mask=None):
    """
    Write an '.xyzin' file, i.e. xyz formatted with some box information so it can be read by LAMMPS.

    Args:
        fout (str): Name of file to write to.
        pos (np.ndarray): :math:`(n,3)` Cartesian atomic positions to write.
        lx (float): Periodic length in x-direction.
        ly (float): Periodic length in y-direction. (Default, None, uses `lx` value.)
        lz (float): Periodic length in z-direction. (Default, None, uses `lx` value.)
        species (np.ndarray): :math:`(n,)` *int* species identifiers. (Default, None, is all species 1.)
        nspecies (int): Number of species (must be *at least* as many as unique species in species array). (Default,
                        None, counts the unique values in `species`.)
        comment (str): Comment string to put at head of file. Must be a single line or the file may not be read
                       properly later. (Default is "Written with clustergb".)
        mask (np.ndarray): :math:`(n,)` *bool* mask indicating which atoms should be written if only a subset is
                           desired. (Default, None, writes the entire array of atoms.)
    """

    if species is None:
        species = np.ones(len(pos), dtype=int)

    ntypes = len(np.unique(species))
    if nspecies is not None:
        if nspecies < ntypes:
            raise ValueError('nspecies must be >= the number of unique species.')
        ntypes = nspecies

    if ly is None:
        ly = lx
    if lz is None:
        lz = lx

    natoms = len(pos)
    ids = np.arange(natoms) + 1

    if mask is not None:
        pos = pos[mask]
        species = species[mask]
        ids = ids[mask]
        natoms = len(pos)

    comment = '#' + comment.rstrip('\n')

    header_str = '{comment}\n\n{natoms} atoms\n{ntypes} atom types\n\n0 {xmax} xlo xhi' \
                     '\n0 {ymax} ylo yhi\n0 {zmax} zlo zhi\n\nAtoms\n'
    context = {'comment': comment, 'natoms': natoms, 'ntypes': ntypes, 'xmax': lx, 'ymax': ly, 'zmax': lz}

    data = np.hstack((ids.reshape(-1, 1), species.reshape(-1, 1), pos))

    np.savetxt(fout, data, header=header_str.format(**context), comments='', fmt='%d %d %f %f %f')


def _parse_xyzin_box_lines(line):
    """Parses lines looking like "22.5 122.5 xlo xhi" to get the span (120 in the example)."""
    data = line.split()
    start = float(data[0])
    end = float(data[1])
    return end - start
