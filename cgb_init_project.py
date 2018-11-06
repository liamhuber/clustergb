#!/usr/bin/env python
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Sets up a Project to host multiple GBs (Jobs) using a particular potential all treated in a consistent manner.

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

    # Make a new project
    obj = cgb.project.Project(args)
    central_id = obj.run_bulk(args.procs)
    for s in obj.par.chem.solutes:
        obj.run_solute(s, central_id, args.procs)

    end = time.time()
    cgb.osio.tee(obj.log_loc, "Project initialization runtime = " + str(end - start) + "s.")


def _ret_parser():
    parser = argparse.ArgumentParser(description="Sets up a Project to host multiple GBs (Jobs) using a particular "
                                                 "potential all treated in a consistent manner.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Project name
    parser.add_argument("name", type=str, help="Unique project name.")

    # Chemistry
    chem = parser.add_argument_group("Chemistry", "Parameters controlling the chemistry.")
    chem.add_argument("--potential", "-potl", type=cgb.lammps.potential_check, required=True,
                      help="Path to the empirical potential to use.")
    chem.add_argument("--host", "-host", type=str, required=True,
                      help="Chemical symbol for host species.")
    chem.add_argument("--solutes", "-sol", nargs="+", default=[],
                      help="Elemental symbol(s) for solute species.")

    # Crystal structure
    xl = parser.add_argument_group("Crystal", "Parameters for the bulk crystal lattice.")
    lattice_args = xl.add_mutually_exclusive_group(required=True)
    lattice_args.add_argument("--fcc", "-fcc", type=float,
                              help="Use fcc lattice with given size.")
    lattice_args.add_argument("--bcc", "-bcc", type=float,
                              help="Use bcc lattice with given size.")
    lattice_args.add_argument("--hcp", "-hcp", type=float, nargs=2,
                              help="Use hcp lattice with given a- and c-lattice length. **Warning:** Not implemented.")

    # Cluster geometry
    cluster = parser.add_argument_group("Cluster geometry", "Parameters for the geometry of the vacuum cluster.")
    cluster.add_argument("--radius", "-r", type=float, default=140.,
                         help="The radius of the cluster.")
    cluster.add_argument("--inner-radius", "-rin", type=float, default=80.,
                         help="Radius of the sub-sphere used for calculating GB energies, finding GB sites, etc.")
    cluster.add_argument("--outer-radius", "-rout", type=float, default=None,
                         help="A slightly larger sub-sphere, for moving a few extra atoms to the head of the "
                              "structure file. Default is the smaller of the inner radius +5 angstroms, or the total "
                              "cluster radius.")
    cluster.add_argument("--vacuum", "-vac", type=float, default=20.,
                         help="The minimum vacuum between periodic images of the cluster.")

    # Macroscopic DoF finding (using Newton's method)
    newton = parser.add_argument_group("Newton's method",
                                       "Parameters for using Newton's method to find bravais lattices for the two "
                                       "grains which match the macroscopic GB degrees of freedom.")
    newton.add_argument("--newton-precision", "-nprec", type=float, default=1E-8,
                        help="Numeric precision limit, mostly for L2-norm difference determining whether two vectors "
                             "are the same. Used in gradient descent when looking for GB bravais vectors.")
    newton.add_argument("--newton-max-steps", "-nms", type=int, default=5000,
                        help="Maximum number of gradient descent steps to use in search for rotation angles when "
                             "looking for GB bravais vectors.")
    newton.add_argument("--newton-verbosity", "-ndv", type=int, default=0,
                        help="Verbosity interval of the gradient descent search for rotation angles. 0 is silent.")

    # Microscopic DoF optimization (using SPSA)
    spsa = parser.add_argument_group("SPSA",
                                     "Parameters for using Simultaneous Perturbation Stochastic Approximation to "
                                     "(stochastically) minimize the GB microscopic DoF. Default values are taken from "
                                     "Spall's 1998 IEEE or chosen to work for minimizing unrelaxed GB energies on "
                                     "the order of 100-1000 mJ/m^2. The minimization takes place in (translation, "
                                     "merge distance) space, where translation is of one grain relative to the other "
                                     "modulo the bravais lattice, and merging for atoms across the GB plane and is "
                                     "bounded.")
    spsa.add_argument("--spsa-probe-size", "-spsaps", type=float, default=None,
                      help="How far (in angstroms) to step when evaluating the simultaneous perturbation for a "
                           "gradient. Default is 0.01 of the lattice constant.")
    spsa.add_argument("--spsa-gamma", "-spsag", type=float, default=0.101,
                      help="The power at which probe size decays with iterations.")
    spsa.add_argument("--spsa-step-size", "-spsass", type=float, default=0.0001,
                      help="Multiplier for how far to follow the gradient.")
    spsa.add_argument("--spsa-step-offset", "-spsaso", type=float, default=1.,
                      help="Iteration offset for damping the decay of the step size (larger means slower decay).")
    spsa.add_argument("--spsa-alpha", "-spsaa", type=float, default=0.602,
                      help="The power at which the step size decays with iterations.")
    spsa.add_argument("--spsa-convergence-distance", "-spsacd", type=float, default=0.1,
                      help="The maximum change in (translation, merge)-space (angstroms) between consecutive minima "
                           "to be considered converged.")
    spsa.add_argument("--spsa-convergence-energy", "-spsace", type=float, default=0.1,
                      help="The maximum change in unrelaxed GB energy (mJ/m^2) between consecutive minima to be "
                           "considered converged")
    spsa.add_argument("--spsa-momentum", "-spsam", type=float, default=0.8,
                      help="Momenta for the stochastic gradient descent, i.e. fraction of last iteration's step to "
                           "add to the current step.")
    spsa.add_argument("--spsa-max-steps", "-spsams", type=int, default=200,
                      help="Maximum number of iterations to use.")
    spsa.add_argument("--spsa-trials", "-spsat", type=int, default=5,
                      help="Number of (semi)randomly initialized SPSA trials to perform.")
    spsa.add_argument("--spsa-verbose", "-spsav", action="store_true",
                      help="Flag to activate verbose output for the SPSA search.")
    spsa.add_argument("--merge-limits", "-merge", nargs=2, metavar=("min", "max"), default=None,
                      help="SPSA searches over a merge distance, over which atoms from different grains will be merged "
                           "across the GB plane. The minimum and maximum distances in the search will "
                           "asymptotically approach the limits given here. Default is one third the 1NN distance to "
                           "85 percent of the NN distance.")

    # LAMMPS parameters
    lammps = parser.add_argument_group("LAMMPS ", "Parameters to control LAMMPS calculations.")
    lammps.add_argument("--lammps-force-convergence", "-ftol", type=float, default=0.001,
                        help="Force convergence (eV/angstrom) for all LAMMPS minimizations.")
    lammps.add_argument("--lammps-max-steps", "-lms", type=int, default=10000,
                        help="Max steps for LAMMPS energy minimization.")
    lammps.add_argument("--lammps-timestep", "-dt", type=float, default=0.002,
                        help="Timestep (ps) for LAMMPS MD.")
    # LAMMPS Annealing properties
    lammps.add_argument("--annealing-time", "-atime", type=float, default=0,
                        help="How long (ps) to anneal for.")
    lammps.add_argument("--annealing-temp", "-atemp", type=float, default=293,
                        help="How hot (K) to anneal.")
    lammps.add_argument("--annealing-init-temp", "-atemp0", type=float, default=None,
                        help="Temperature (K) for initial velocities during annealing. Default is 1.5x temperature.")
    lammps.add_argument("--annealing-langevin", "-alangevin", type=float, default=5.,
                        help="LAMMPS-format Langevin damping parameter (ps) to use for thermostat while annealing.")
    # LAMMPS Quenching properties
    lammps.add_argument("--quenching-samples", "-qsamples", type=int, default=1,
                        help="Number of times to sample the annealed state with a quench and minimization")
    lammps.add_argument("--quenching-damping", "-qdamping", type=float, default=0.01,
                        help="Strength of quench (friction: force/velocity).")
    lammps.add_argument("--quenching-time", "-qtime", type=float, default=1.,
                        help="Duration of quench (ps).")

    # Output
    parser.add_argument("--decompose", "-dcomp", action='store_true',
                        help="In addition to segregation calculations, run pure-host structures with solute-relaxed "
                             "structures so energies can later be decomposed into chemical and deformation components.")

    # Runtime commands
    parser.add_argument("--procs", "-np", type=int, default=1,
                        help="Max processors available to run mpi on.s")

    return parser


if __name__ == "__main__":

    returned_parser = _ret_parser()
    arguments = returned_parser.parse_args()

    main(arguments)
