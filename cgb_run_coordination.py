#!/usr/bin/env python
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
For each grain boundary, calculates the Fermi-smeared coordination number of each GB site.

If used, please cite `Huang *et al.*, PSSB 252 (2015) <bibtex_refs/huang2015improvement.html>`_
"""

from __future__ import absolute_import
import argparse
#from . import clustergb as cgb
import clustergb as cgb
import numpy as np
from os.path import join
import time

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def run_fermi_smeared_coordination(job, force_recalculate=False, fermi_smearing=None):
    """
    Calculate the coordination number for each GB site at a particular boundary.

    Args:
        job (clustergb.job.Job): Job on which to run.
        force_recalculate (bool): Whether to overwrite existing data (default=False.)
        fermi_smearing (float): Smearing value to use. (Default is None, which triggers the default in the
                                clustergb.coordination module.)
    """
    start = time.time()

    cgb.osio.tee(job.log_loc, "Starting fermi-smeared coordination number calculation for " +job.name)

    # Make sure there is a container available for coordination numbers
    coord = job.ensure_namespace("coordination", scope=job.results)

    # Check if this GB already has undecorated coordination values
    if hasattr(coord, "number") and not force_recalculate:
        return

    # Set the fermi smearing parameters
    center, fermi_smearing = cgb.coordination.auto_parameterize(job.par.xl.nn_dist, job.par.xl.snn_dist, fermi_smearing)

    # Read the structure and keep all the atoms inside outer_r
    pos, _, _, _, _ = cgb.structure.read_xyzin(join(job.location, "gb.xyzin"))
    pos = pos[:-job.gb.natoms_balance]  # Trim the fat for computational efficiency

    # Get the coordination of GB sites
    targets = np.arange(job.gb.natoms_nonbulklike)
    number, closest = cgb.coordination.coordination(pos, center, fermi_smearing, indices=targets)

    # Save what we've done
    coord.center = center
    coord.smearing = fermi_smearing
    setattr(coord, "number", number)
    setattr(coord, "closest", closest)
    job.save()

    end = time.time()
    cgb.osio.tee(job.log_loc, job.name + " coordination number runtime = " + str(end - start) + "s.")


def main(args):
    start = time.time()

    cgb.osio.run_in_hierarchy(run_fermi_smeared_coordination, vars(args))

    end = time.time()
    print("Total coordination number runtime = " + str(end - start) + "s.")


def _ret_parser():
    parser = argparse.ArgumentParser(description="Calculate the Fermi-smeared coordination number of each GB site at "
                                                 "each GB recursively through the ClusterGB hierarchy. Saves results "
                                                 "to Job.results.coordination.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--force-recalculate", "-recalc", action="store_true",
                        help="Overwrite any existing coordination number data with new calculations.")

    parser.add_argument("--fermi-smearing", "-smear", default=None,
                        help="Amount (angstroms) of fermi-smearing to apply when calculating coordination values. "
                             "(Default is 20 percent of the difference between first and second nearest neighbour "
                             "distances.)")
    return parser


if __name__ == "__main__":
    returned_parser = _ret_parser()

    arguments = returned_parser.parse_args()

    main(arguments)
