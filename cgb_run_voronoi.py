#!/usr/bin/env python
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
For each grain boundary, calculates the Voronoi volume and other Voronoi features for each GB site.

If used, please cite `whatever Voro++ tells you <http://math.lbl.gov/voro++/download/>`_
"""

from __future__ import absolute_import
import argparse
#from . import clustergb as cgb
import clustergb as cgb
import os
import time

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def run_voronoi_volume(job, force_recalculate=False):
    """
    Calculate the Voronoi properties for each GB site at a particular boundary using Voro++.

    Args:
        job (clustergb.job.Job): Job on which to run.
        force_recalculate (bool): Whether to overwrite existing data (default=False.)
    """
    start = time.time()

    cgb.osio.tee(job.log_loc, "Starting Voronoi volume calculation for " + job.name)

    # Make sure there is a container available for Voronoi volumes
    voro = job.ensure_namespace("voronoi", scope=job.results)

    # Check if this GB already has undecorated Voronoi volumes
    if hasattr(voro, "volume") and not force_recalculate:
        return

    # Run voro++
    pos, _, lx, ly, lz = cgb.structure.read_xyzin(os.path.join(job.location, "gb.xyzin"))
    volume, area, edge_length, vertices, faces, edges, offset = cgb.voronoi.voronoi(pos, lx, ly, lz)

    # Save results
    setattr(voro, "volume", volume[:job.gb.natoms_nonbulklike])
    setattr(voro, "area", area[:job.gb.natoms_nonbulklike])
    setattr(voro, "edge_length", edge_length[:job.gb.natoms_nonbulklike])
    setattr(voro, "vertices", vertices[:job.gb.natoms_nonbulklike])
    setattr(voro, "faces", faces[:job.gb.natoms_nonbulklike])
    setattr(voro, "edges", edges[:job.gb.natoms_nonbulklike])
    setattr(voro, "offset", offset[:job.gb.natoms_nonbulklike])
    job.save()

    end = time.time()
    cgb.osio.tee(job.log_loc, job.name + " Voronoi runtime = " + str(end - start) + "s.")


def main(args):
    start = time.time()

    cgb.osio.run_in_hierarchy(run_voronoi_volume, vars(args))

    end = time.time()
    print("Total Voronoi runtime = " + str(end - start) + "s.")


def _ret_parser():
    parser = argparse.ArgumentParser(description="Calculate the Voronoi properties of each GB site at each GB "
                                                 "recursively through the ClusterGB hierarchy.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--force-recalculate", "-recalc", action="store_true",
                        help="Overwrite any existing Voronoi data with new calculations.")
    return parser


if __name__ == "__main__":
    returned_parser = _ret_parser()

    arguments = returned_parser.parse_args()

    main(arguments)
