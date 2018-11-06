#!/usr/bin/env python
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Initialize a cluster containing a GB with particular macroscopic character and minimize the microscopic degrees of
freedom according to parameters from the parent Project.

Please cite:

- `Lee and Choi, MSMSE 12 (2004) <bibtex_refs/lee2004computation.html>`_
- `Huber et al., in preparation (2018) <bibtex_refs/huber2018prep.html>`_
"""

from __future__ import absolute_import
import argparse
#from . import clustergb as cgb
import clustergb as cgb
import time

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def main(args):
    start = time.time()

    job = cgb.job.Job(args)
    job.new_boundary(args.procs)

    end = time.time()
    cgb.osio.tee(job.log_loc, "Job initialization runtime = " + str(end - start) + "s.")
    return


def _ret_parser():
    parser = argparse.ArgumentParser(description="Initialize a cluster containing a GB with particular macroscopic "
                                                 "character and minimize the microscopic degrees of freedom according "
                                                 "to parameters from the parent Project.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Runtime commands
    # Required
    parser.add_argument("name", type=str, help="Unique job name.")
    # Has default
    parser.add_argument("--procs", "-np", type=int, default=1,
                        help="Max processors available to run mpi on.")
    parser.add_argument("--debug", "-dbug", action="store_true",
                        help="Rename LAMMPS log files so you can see all of them.")

    # Grain boundary settings
    subparsers = parser.add_subparsers(help="Define the macroscopic degrees of freedom for the GB.", dest="gb_style")

    gb_parser = subparsers.add_parser("macro", help="Define the GB by providing shared and normal axes along with a "
                                                    "misorientation.")

    gb_parser.add_argument("--misorientation", "-m", type=float, default=None,
                           help="Relative misorientation between two grains in degrees. (Default is random value.)")
    gb_parser.add_argument("--shared", "-s", type=float, default=None, nargs=3,
                           help="Shared axis for GB. (Default is random.)")
    normal_args = gb_parser.add_mutually_exclusive_group()
    #normal_args.add_argument("--symmetric", "-sym", action="store_true",
    #                         help="Define the GB normal vector by symmetrically rotating both grains by half "
    #                              "of the misorientation angle. Makes a pure-tilt boundary.")
    # Symmetric tilt boundaries are not unique for a given misorientation. Leave the code for doing this, but don't
    # expose it to the user at the moment.
    normal_args.add_argument("--normal", "-n", type=float, default=None, nargs=3,
                             help="GB plane normal vector for one grain. Should be normal to shared axis. (Default is "
                                  "random.)")

    brav_parser = subparsers.add_parser("rot", help="Define the GB by providing two rotation matrices (row by row, "
                                                    "2*3*3 = 18 elements total). This is the description used by "
                                                    "Olmsted, Foiles, and Holm (2009) and many boundaries can be "
                                                    "found in their supplementary material.")
    brav_parser.add_argument("rot1", nargs=9, type=float, help="Rotation matrix for the first grain.")
    brav_parser.add_argument("rot2", nargs=9, type=float,
                             help="Rotation matrix for the second grain. Must also share one vector with the first "
                                  "rotation matrix")

    copy_parser = subparsers.add_parser("copy", help="Simply copy the Bravais lattice of each grain from another Job.")
    copy_parser.add_argument("job_dir", type=str,  help="Path to directory where another Job exists")

    # Output settings
    output_parser = parser.add_argument_group("Output", "Parameters for output frequency.")
    output_parser.add_argument("--thermo-period", "-thermo", type=int, default=None,
                               help="Period for writing thermodynamic output during annealing.")
    output_parser.add_argument("--dump-period", "-dump", type=int, default=None,
                               help="Period for dumping xyz positions during annealing. (Default is first and last "
                                    "only.)")
    return parser

if __name__ == "__main__":

    returned_parser = _ret_parser()
    arguments = returned_parser.parse_args()

    main(arguments)


