#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Coordination number (number of neighbours inside a cutoff) is a popular metric for structures. This code implements the
Fermi-smeared version of this value from [1]_ which is more appropriate for the chaotic GB environment.

.. [1] Huang, Grabowski, McEniry, Trinkle, Neugebauer, Phys. Status Solidi B 252 (2015)
"""

import numpy as np
from .extra_math import fermi

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def auto_parameterize(nn_dist, snn_dist, smear=None):
    """
    Automatically calculate fermi parameters from crystal properties so that the midpoint and width of the smearing
    depend on the distance between first and second nearest neighbours.

    Args:
        nn_dist (float): Nearest neighbour distance.
        snn_dist (float): Second-nearest neighbour distance.
        smear (float): Fermi sigma value to use. (Default is twenty percent of the first and second nearest neighbour
                       distance difference.)

    Returns:
        2-element tuple containing

        - (*float*) -- Distance for half contribution.
        - (*float*) -- Smearing width.
    """
    center = 0.5 * (nn_dist + snn_dist)
    if smear is None:
        # Set smearing to 20% of the 2NN-1NN distance
        percentage = 0.2
        smear = percentage * (snn_dist - nn_dist)
    return center, smear


def coordination(pos, r0, sigma, indices=None):
    """
    Calculates the Fermi-smeared coordination numbers for a structure as

    .. math::

        N_i = \\sum_j \\exp \\left( \\frac{r_{ij} - r_0}{\sigma} \\right).

    Args:
        pos (np.ndarray): :math:`(n, 3)` vector of atomic positions to use when calculating coordination.
        r0 (float): Distance at which an atom contributes 0.5 to coordination.
        sigma (float): Fermi smearing parameter for transitioning from coordination 1 down to 0.
        indices (np.ndarray or int): Specific atoms in `pos` to calculate the coordination of. (Default is all the
                                     atoms.)
    Returns:
        2-element tuple containing

        - (*np.ndarray*) -- Smeared coordination numbers.
        - (*np.ndarray*) -- Distances to closest neighbour.

    .. todo::

        Add a cutoff distance so there's atomic binning and we get :math:`O(N)` scaling instead of :math:`O(N^2)`.
    """
    if indices is None:
        indices = np.arange(len(pos))

    try:
        n_targets = len(indices)
    except TypeError:
        n_targets = 1
        indices = np.array([indices])

    number = np.empty(n_targets)
    closest = np.empty(n_targets)

    fermi0 = fermi(0, r0, sigma)  # Self-coordination
    for i, ind in enumerate(indices):
        disp = pos - pos[ind]
        dist_sq = np.sum(disp * disp, axis=1)
        dist = np.sqrt(dist_sq)
        fermir = fermi(dist, r0, sigma)
        number[i] = np.sum(fermir) - fermi0
        closest[i] = np.min(np.delete(dist, ind))

    return number, closest
