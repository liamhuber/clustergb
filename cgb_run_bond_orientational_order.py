#!/usr/bin/env python
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
For each grain boundary, calculates the bond-order parameters of each GB site.

Assumes a cutoff between the second and third nearest neighbours.

If used, please cite Steinhardt, Nelson, and Ronchetti, PRB 28 (1983).

.. WARNING: If the number of allowed crystal structure in the main code are extended, you'll need to extend the
            calculation of the cutoff radius here too.
"""

from __future__ import absolute_import
import argparse
#from . import clustergb as cgb
import clustergb as cgb
import os
import time
import numpy as np

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def run_bond_orientational_order(job, force_recalculate=False, max_order=8):
    """
    Calculate the bond orientational order parameters for each GB site at a particular boundary.

    Args:
        job (clustergb.job.Job): Job on which to run.
        force_recalculate (bool): Whether to overwrite existing data (default=False.)
        max_order (int): Maximum order of spherical harmonics to use.
    """
    start = time.time()

    cgb.osio.tee(job.log_loc, "Starting bond-order calculation for " + job.name)

    # Make sure there is a container available for bond order parameters
    bondo = job.ensure_namespace("bond_order", scope=job.results)

    # Check if this GB already has the largest Q and we're not recalculating, just pass
    if hasattr(bondo, "q" + str(max_order)) and not force_recalculate:
        return

    # Run bond order calculation
    pos, _, lx, ly, lz = cgb.structure.read_xyzin(os.path.join(job.location, "gb.xyzin"))
    xl = job.par.xl.type
    gb_ids = np.arange(job.gb.natoms_nonbulklike)
    latt = job.par.xl.length
    qs = cgb.bond_orientational_order.bond_orientational_order_parameter(pos, gb_ids, xl, latt, lmax=max_order)

    # Save results in our job
    for l in np.arange(max_order):
        setattr(bondo, "q" + str(l + 1), qs[:, l])
    job.save()

    end = time.time()
    cgb.osio.tee(job.log_loc, job.name + " bond order runtime = " + str(end - start) + "s.")


def main(args):
    start = time.time()

    cgb.osio.run_in_hierarchy(run_bond_orientational_order, vars(args))

    end = time.time()
    print("Total bond-order runtime = " + str(end - start) + "s.")


def _ret_parser():
    parser = argparse.ArgumentParser(description="Calculate the bond-orientational order parameters of each GB site at "
                                                 "each GB recursively through the ClusterGB hierarchy.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--force-recalculate", "-recalc", action="store_true",
                        help="Overwrite any existing bond-orientational order parameter data with new calculations.")
    parser.add_argument("--max-order", "-maxo", default=8,
                        help="Maximum order (`l` in `Y_l^m` spherical harmonics) to calculate.")
    parser.add_argument("--cutoff", "-cut", default=None,
                        help="Cutoff distance to include neighbours in bond counting. Default is 3rd nearest "
                             "neighbour distance.")
    return parser


if __name__ == "__main__":
    returned_parser = _ret_parser()

    arguments = returned_parser.parse_args()

    main(arguments)
