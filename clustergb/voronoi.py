#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Run `voro++ <http://math.lbl.gov/voro++/>`_ on a set of positions using the voro++ executable listed in the config file.
"""

import os
import yaml
from .structure import write_vorin
from .osio import run_in_shell
from numpy import genfromtxt

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def voronoi(pos, lx, ly, lz):
    """
    Given a set of n atomic positions in a rectangular prismatic cell, calculates their Voronoi volumes using voro++.

    .. warning::

        All the positions should lie inside the cell, and the calculation is aperiodic, so the cell walls function as
        Voronoi cell boundaries.

    .. todo::

        Why have a warning when you can just force it to be true by operating on `pos`?

    Args:
        pos (np.ndarray): :math:`(n,3)` Cartesian atomic positions to evaluate.
        lx, ly, lz (float): x-, y-, and z-lengths for rectangular cell.

    Returns:
            7-element tuple containing

            - (*np.ndarray*) -- :math:`(n,)` Voronoi volumes.
            - (*np.ndarray*) -- :math:`(n,)` surface areas.
            - (*np.ndarray*) -- :math:`(n,)` edge lengths.
            - (*np.ndarray*) -- :math:`(n,)` *int* vertex counts.
            - (*np.ndarray*) -- :math:`(n,)` *int* face counts.
            - (*np.ndarray*) -- :math:`(n,)` *int* edge counts.
            - (*np.ndarray*) -- :math:`(n,3)` offset of the Voronoi centroid from the atom for which it was constructed.
    """

    # Read the executable name
    config_file_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(config_file_dir, "config.yml"), "r") as f:
        config = yaml.load(f)
    voro_exec = config["executables"]["voropp"]

    # Write a file for voro++ to read
    vorin = "tmp.vorin"
    write_vorin(vorin, pos)

    # Run voro++
    command = voro_exec + " -c '%v %F %E %w %s %g %c' -o " + " ".join(str(l) for l in [0, lx, 0, ly, 0, lz]) + " " + \
              vorin
    run_in_shell(command)

    # Parse the results
    vorout = vorin + ".vol"
    voro_data = genfromtxt(vorout)
    volume = voro_data[:, 0]
    area = voro_data[:, 1]
    edge_length = voro_data[:, 2]
    vertices = voro_data[:, 3].astype(int)
    faces = voro_data[:, 4].astype(int)
    edges = voro_data[:, 5].astype(int)
    offset = voro_data[:, 6:]

    # Clean up
    os.remove(vorin)
    os.remove(vorout)

    return volume, area, edge_length, vertices, faces, edges, offset
