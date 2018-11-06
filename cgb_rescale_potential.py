#!/usr/bin/env python
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Rescales a potential to give a target 0 K lattice parameter and writes the modified potential to a file.

This isn't an integral part of the clustergb package, but is extremely likely to be useful since it lets you match the
lattice constants of multiple EAM potentials, allowing (as best as possible) an apples-to-apples comparison of
different potentials.
"""

from __future__ import absolute_import
import argparse
import os
import shutil
import numpy as np
#from . import clustergb as cgb
import clustergb as cgb

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def main(args):

    targ_latt = args.fcc or args.bcc
    xl_type = np.array(['fcc', 'bcc'])[np.array([args.fcc, args.bcc], dtype=bool)][0]

    last_dr = read_dr(args.potential)
    last_err = get_lattice_difference(xl_type, targ_latt, args.potential, args.species, args.strain, args.nsamples,
                                      procs=args.procs, max_steps=args.lammps_max_steps,
                                      force_convergence=args.lammps_force_convergence)

    if args.check_only:
        print("Energy vs strain fit gives an optimal lattice parameter of " + str(targ_latt + last_err) + " for this "
              "potential. Terminating after lattice check.")
        return

    if abs(last_err) < args.precision:
        actual_latt = targ_latt + last_err
        print("The potential already has a lattice constant of " + str(actual_latt) + " which is within "
              "the requested precision. No new potential file written.")

    # Copy the potential to a new rescaled file
    base_name = os.path.basename(os.path.realpath(args.potential))
    pot_dir = os.path.dirname(os.path.realpath(args.potential))

    # Slice and dice the name to insert 'rescaled_x_y' into the filename where x.y is the target lattice constant
    latt_string = '_'.join(str(targ_latt).split('.'))
    split_base = base_name.split('.')
    name_components = [split_base[0] + '_rescaled_' + latt_string]
    name_components += [comp for comp in split_base[1:]]
    rescaled_pot_name = '.'.join(name_components)  # Format the string all properly

    rescaled_pot = os.path.join(pot_dir, rescaled_pot_name)
    shutil.copyfile(args.potential, rescaled_pot)

    # Rescale the copied potential
    dr = last_dr + args.init_dr_step
    set_dr(rescaled_pot, dr)

    # Get the error with the new rescaled potential
    err = get_lattice_difference(xl_type, targ_latt, rescaled_pot, args.species, args.strain, args.nsamples,
                                 procs=args.procs, max_steps=args.lammps_max_steps,
                                 force_convergence=args.lammps_force_convergence)

    print("Step, error")
    step = 0
    print(step, err)

    while abs(err) > args.precision:
        new_dr = dr - err * (dr - last_dr) / (err - last_err)  # Secant method
        set_dr(rescaled_pot, new_dr)
        new_err = get_lattice_difference(xl_type, targ_latt, rescaled_pot, args.species, args.strain, args.nsamples,
                                         procs=args.procs, max_steps=args.lammps_max_steps,
                                         force_convergence=args.lammps_force_convergence)

        last_err = err
        last_dr = dr
        err = new_err
        dr = new_dr
        step += 1
        print(step, err)

    print("Rescaled potential written to " + rescaled_pot + ".")
    print("The final lattice constant was " + str(targ_latt + err) + ".")

    return


def get_lattice_difference(xl_type, targ_latt, potential, species, strain, samples,
                           max_steps=1000, force_convergence=0.0001, procs=1):
    """
    Relax a unit cell at zero pressure to get an approximation of the minimum lattice constant, then run a series of
    strained static calculations and fit a quadratic to the energy curve to get a more accurate minimum. Return the
    difference between the target lattice `targ_latt` and the actual lattice returned by the `potential`.

    Args:
        xl_type (str): Crystal structure indicator.
        targ_latt (float): Target lattice constant and initial guess for isobaric relaxation.
        potential (str): Path to a valid EAM potential to use.
        species (str): Chemical symbol of the species to use from the potential.
        strain (float): Maximum strain (as a fraction of optimal lattice parameter) for the energy-volume curve.
        samples (int): Number of data points on the energy-volume curve.
        max_steps (int):  Maximum number of conjugate gradient steps to take.
        force_convergence (float): Stopping threshold based on the L2 norm of the global force vector.
        procs (int): How many processors to run on. (Default is 1.)

    Returns:
        (*float*) -- The difference between minimum energy lattice constant and the target value.
    """

    # Run minimization to get estimate of min value
    data = cgb.lammps.run_minimization_bulk(xl_type, targ_latt, potential, species,
                                            max_steps=max_steps, force_convergence=force_convergence, nprocs=procs)
    init_latt = data.lx


    # Do a strain about that minimum value
    min_strained = init_latt * (1 - strain)
    max_strained = init_latt * (1 + strain)
    strained_latts = np.linspace(min_strained, max_strained, samples)

    _, energies = cgb.lammps.energy_volume_curve(xl_type, strained_latts, potential, species, nprocs=procs)

    # Fit a quadratic to the energy and get the minima
    fit = np.polyfit(strained_latts, energies, 2)
    min_latt = -fit[1] / (2 * fit[0])

    return min_latt - targ_latt


def read_dr(potential):
    """Read the `dr` value (real-space stepping of the tabulated values) of an EAM potential."""
    pot_type = cgb.osio.file_extension(potential)

    if pot_type == 'alloy':
        return read_dr_alloy(potential)
    elif pot_type == 'fs':
        return read_dr_finnis_sinclair(potential)
    else:
        raise Exception('Unrecognized potential type, "' + pot_type + '".')


def read_dr_alloy(potential):
    with open(potential, 'r') as fpot:
        contents = fpot.readlines()
    dr = float(contents[4].split()[3])
    return dr


def read_dr_finnis_sinclair(potential):
    return read_dr_alloy(potential)


def set_dr(potential, new_dr):
    """Change the `dr` value (real-space stepping of the tabulated values) for an EAM potential."""
    pot_type = cgb.osio.file_extension(potential)

    if pot_type == 'alloy':
        set_dr_alloy(potential, new_dr)
    elif pot_type == 'fs':
        set_dr_finnis_sinclair(potential, new_dr)
    else:
        raise Exception('Unrecognized potential type, "' + pot_type + '".')
    return


def set_dr_alloy(potential, new_dr):
    with open(potential, 'r') as fpot:
        contents = fpot.readlines()
    data = contents[4].split()
    data[3] = str(new_dr)
    contents[4] = ' '.join(data) + '\r\n'
    with open(potential, 'w') as fpot:
        fpot.writelines(contents)
    return


def set_dr_finnis_sinclair(potential, new_dr):
    set_dr_alloy(potential, new_dr)


def _ret_parser():
    parser = argparse.ArgumentParser(description='Rescale an EAM potential to match a target lattice parameter at 0K.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Material parameters
    parser.add_argument('potential', type=cgb.lammps.potential_check,
                        help='Path to the empirical potential to rescale.')
    parser.add_argument('species', type=str,
                        help='Elemental symbol for the host (as referenced in the potential.)')

    lattice_args = parser.add_mutually_exclusive_group(required=True)
    lattice_args.add_argument('--fcc', '-fcc', type=float,
                              help='Use fcc lattice and target given size (in angstroms).')
    lattice_args.add_argument('--bcc', '-bcc', type=float,
                              help='Use bcc lattice and target given size (in angstroms).')

    # Numeric parameters
    parser.add_argument('--precision', '-prec', type=float, default=0.00001,
                        help='Precision for matching actual lattice constant to target.')
    parser.add_argument('--strain', '-stn', type=float, default=0.00001,
                        help='Maximum strain away from target value for energy-volume curve.')
    parser.add_argument('--nsamples', '-n', type=int, default=8,
                        help='Number of points in energy-volume curve.')
    parser.add_argument('--init_dr_step', '-dr0', type=float, default=0.00001,
                        help='Initial step for changing the potential\'s dr value.')
    parser.add_argument('--lammps_force_convergence', '-ftol', type=float, default=0.001,
                        help='Force convergence for all LAMMPS minimizations in eV/angstrom.')
    parser.add_argument('--lammps_max_steps', '-lms', type=int, default=10000,
                        help='Max steps for LAMMPS energy minimization.')
    parser.add_argument('--procs', '-np', type=int, default=1,
                        help='Max processors available to run mpi on.')

    # Alternate run-style
    parser.add_argument('--check_only', '-check', action='store_true',
                        help='Instead of rescaling the potential, just get its optimal lattice constant from an '
                             'energy-volume curve.')
    return parser


if __name__ == '__main__':

    returned_parser = _ret_parser()
    arguments = returned_parser.parse_args()

    main(arguments)