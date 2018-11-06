#!/usr/bin/env python
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Calculate and print the bulk modulus (GPa) and volume per atom (cubic angstroms) -- as a function of cell size (if more
than one unit cell repetitions were requested).
"""

from __future__ import absolute_import
import argparse
#from . import clustergb as cgb
import clustergb as cgb
import numpy as np

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def main(args):
    finput = 'in.pressure.lammps'
    evang3_to_gpa = 160.21766

    latt_guess = args.fcc or args.bcc
    xl_type = np.array(['fcc', 'bcc'])[np.array([args.fcc, args.bcc], dtype=bool)][0]
    n_unit = np.array([4, 2])[np.array([args.fcc, args.bcc], dtype=bool)][0]

    if args.solute is not None:
        print(args.solute + " in " + args.host)
        print("Repetitions, atom count, B (GPa), host vol, sol vol, sol pressure")
    else:
        print(args.host)
        print("Repetitions, atom count, B (GPa), host vol")

    if args.repetition_samples < 2:
        rep_range = [args.repetitions]
    else:
        rep_range = np.logspace(0, np.log2(args.repetitions), args.repetition_samples, base=2).astype(int)
        rep_range[-1] = args.repetitions

    for reps in rep_range:

        n_total = n_unit * reps**3

        V0, B = _bulk_mod(xl_type, latt_guess, args.potential, args.host, args.lammps_max_steps,
                          args.lammps_force_convergence, finput, args.procs, args.strain, args.nsamples, reps)

        vol_per_atom = V0 / float(n_total)

        if args.solute is not None:
            V0x, Bx = _bulk_mod(xl_type, latt_guess, args.potential, args.host, args.lammps_max_steps,
                                args.lammps_force_convergence, finput, args.procs, args.strain, args.nsamples,
                                reps, solute=args.solute, solute_ids=0)
            sol_vol = V0x - V0 + vol_per_atom
            sol_press = B * (V0x - V0) / vol_per_atom
            print(reps, n_total, B * evang3_to_gpa, vol_per_atom, sol_vol, sol_press)
        else:
            print(reps, n_total, B * evang3_to_gpa, vol_per_atom)

    return


def _bulk_mod(xl_type, latt_guess, pot_file, species, max_steps, force_convergence, input_file, nprocs,
              strain, nsamples, cell_repetitions, solute=None, solute_ids=None):
    """
    Using repetitions of the unit cell, run a relaxation of the cell size to get a good guess for the optimal lattice
    constant, then calculate an energy-volume curve by straining about this. Extrace ideal cell volume (for supercell
    if more than one repetition of the unit cell is used) and bulk modulus. Can also accommodate a single
    substitutional solute atom.

    Args:
        xl_type (str): Crystal structure identifier.
        latt_guess (float): Best-guess for ideal lattice constant.
        pot_file (str): Path to empirical potential to use.
        species (str): Space separated, potential-appropriate chemical symbols who will be applied (in order) to
                       the type column of the input file.
        max_steps (int): Maximum number of CG steps to take.
        force_convergence (float): Stopping threshold based on the L2 norm of the global force vector.
        input_file (str): File to write LAMMPS input script to.
        nprocs (int): How many processors to run on.
        strain (float): Max fraction of the ideal lattice parameter to strain for creating energy-volume curve.
        nsamples (int): Number of points on energy-volume curve.
        cell_repetitions (int): How many repetitions of the unit cell to allow (in each of x-, y-, and
                                z-directions.)
        solute: solute (str): Chemical symbol for the solute species. (Default is None.)
        solute_ids (int or np.ndarray or list): Integer id(s) for which to change species. (Default is None.)

    Returns:
        2-element tuple containing

        - (*float*) Ideal volume.
        - (*float*) Bulk modulus (in GPa).
    """
    if solute is not None:
        species += ' ' + solute
    min_data = cgb.lammps.run_minimization_bulk(xl_type, latt_guess, pot_file, species,
                                                cell_repetitions=cell_repetitions, solute_ids=solute_ids,
                                                max_steps=max_steps, force_convergence=force_convergence,
                                                input_file=input_file, nprocs=nprocs)
    better_latt_guess = min_data.lx / cell_repetitions
    min_lat = (1 - strain) * better_latt_guess
    max_lat = (1 + strain) * better_latt_guess

    lattice_constants = np.linspace(min_lat, max_lat, nsamples)

    volumes, energies = cgb.lammps.energy_volume_curve(xl_type, lattice_constants, pot_file, species,
                                                       cell_repetitions=cell_repetitions, solute_ids=solute_ids,
                                                       input_file=input_file, nprocs=nprocs)

    fit = np.polyfit(volumes, energies, 2)
    V0 = -fit[1] / (2 * fit[0])
    B = 2 * V0 * fit[0]
    return V0, B


def _ret_parser():
    parser = argparse.ArgumentParser(description='Calculates the bulk modulus (in GPa) for a given potential, host '
                                                 'species, and crystal lattice by fitting a quadratic curve to the '
                                                 'energy-volume data. A single substitutional solute can be added to '
                                                 'the cell if the potential will allow it. In this case, it is '
                                                 'advisable to increase the cell repetitions and look at the change '
                                                 'in bulk modulus, volume, etc. as a function of supercell size (i.e. '
                                                 'solute concentration).',
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Required
    parser.add_argument('potential', type=cgb.lammps.potential_check,
                        help='Path to the empirical potential to use.')
    parser.add_argument('host', type=str,
                        help='Elemental symbol for the host.')

    lattice_args = parser.add_mutually_exclusive_group(required=True)
    lattice_args.add_argument('--fcc', '-fcc', type=float,
                              help='Use fcc lattice with the given initial guess for lattice constant (in angstroms).')
    lattice_args.add_argument('--bcc', '-bcc', type=float,
                              help='Use bcc lattice with the given initial guess for lattice constant (in angstroms).')

    # Optional
    parser.add_argument('--solute', '-sol', type=str, default=None,
                        help='Elemental symbol for a solute.')
    parser.add_argument('--repetitions', '-reps', type=int, default=1,
                        help='Maximum number of unit cell repetitions to use for cell.')
    parser.add_argument('--repetition_samples', '-rep_samples', type=int, default=1,
                        help='Number of logarithmically sampled cell repetitions to use between 1 and `repetitions`.')
    parser.add_argument('--strain', '-stn', type=float, default=0.00001,
                        help='Maximum strain magnitude for generating energy-volume curve.')
    parser.add_argument('--nsamples', '-n', type=int, default=8,
                        help='Number of points in energy-volume curve.')
    parser.add_argument('--lammps_force_convergence', '-ftol', type=float, default=0.001,
                        help='Force convergence for all LAMMPS minimizations in eV/angstrom. (Minimizations are used '
                             'to find a good guess for the ideal volume before applying strains and using static '
                             'calculations to generate an energy-volume curve.)')
    parser.add_argument('--lammps_max_steps', '-lms', type=int, default=10000,
                        help='Max steps for LAMMPS energy minimization.')
    parser.add_argument('--procs', '-np', type=int, default=1,
                        help='Max processors available to run mpi on.')
    return parser

if __name__ == '__main__':

    returned_parser = _ret_parser()
    arguments = returned_parser.parse_args()

    main(arguments)
