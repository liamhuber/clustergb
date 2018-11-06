#!/usr/bin/env python
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
For each grain boundary, calculates the segregation energy of each solute to each GB site.
"""

from __future__ import absolute_import
import argparse
#from . import clustergb as cgb
import clustergb as cgb
import os
import numpy as np
import time

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def run_monosegregation(job, procs, save_relaxed=False, save_radius=None, force_recalculate=False, solute_voronoi=False,
                        solute_coordination=False, fermi_smearing=None, decompose=False, mod=[None, None]):
    """
    Calculate the segregation energy for all solutes at each GB site.

    Properties of the relaxed structure (in the presence of the solute atom) can be calculated as requested.

    Args:
        job (clustergb.job.Job): Job on which to run.
        procs (int): How many processors to call LAMMPS with.
        save_relaxed (bool): Whether to save relaxed structures. (Default is False.)
        save_radius (float): How far around the solute site to save atoms if the relaxed structure is saved. (Default
                             is to save the entire structure.)
        force_recalculate (bool): Whether to overwrite existing data (default=False.)
        solute_voronoi (bool): Whether to calculate Voronoi properties for the relaxed solute. (Default is False.)s
        solute_coordination (bool): Whether to calculate the coordination number for the relaxed solute. (Default is
                                    False.)
        fermi_smearing (float): Smearing value to use for the coordination number. (Default is None, which triggers the
                                default in the clustergb.coordination module.)
        decompose (bool): Whether to run the extra calculations to decompose energies into  "mechanical" and "chemical"
                          components. (Default is False.)
        mod (list): Two *int* values, calculate segregation energy only for sites whose id % `mod[1]` = `mod[0]`.
                    (Default is [None, None], which runs for all sites.)
    """
    start = time.time()

    cgb.osio.tee(job.log_loc, "Starting monosegregation calculation for " + job.name)

    job.procs = procs
    os.chdir(job.location)
    cgb.osio.make_dir("monosegregation", allow_exists=True)
    os.chdir("monosegregation")

    job.refresh_project()  # Make sure you have the most up-to-date project data in case another child started decomp

    n_sites = job.gb.natoms_nonbulklike
    monoseg = job.ensure_namespace("monosegregation", scope=job.results)
    decompose = decompose or job.project.par.output.decompose
    need_pos = force_recalculate or solute_voronoi or solute_coordination or decompose

    # Check if a modulo has been requested, if yes we'll only run the calculation of sites whose id matches the mod
    mod_n = None
    mod_m = None
    use_mod = (mod[0] is not None) and (mod[1] is not None)
    if use_mod:
        mod_n = mod[0]
        mod_m = mod[1]

    # For each solute species and each site, calculate the segregation
    for n, sol in enumerate(job.par.chem.solutes):
        cgb.osio.tee(job.log_loc, "Calculating " + sol + " binding to " + job.name + "...")
        cgb.osio.make_dir(sol, allow_exists=True)
        os.chdir(sol)
        sol_dir = os.getcwd()

        # Make sure there's a space for the solute-specific results and reset them if forcing recalculation
        sol_results = job.ensure_namespace(sol, scope=monoseg)
        if force_recalculate or not hasattr(sol_results, "energy"):
            sol_results.energy = np.ones(n_sites) * np.nan

        # Similarly for voronoi
        if solute_voronoi:
            l = job.par.cluster.size
            voro = job.ensure_namespace("voronoi", scope=sol_results)
            if force_recalculate or not hasattr(voro, "volume"):
                voro.volume = np.ones(n_sites) * np.nan
                voro.area = np.ones(n_sites) * np.nan
                voro.edge_length = np.ones(n_sites) * np.nan
                voro.vertices = np.ones(n_sites, dtype=int) * np.nan
                voro.faces = np.ones(n_sites, dtype=int) * np.nan
                voro.edges = np.ones(n_sites, dtype=int) * np.nan
                voro.offset = np.ones((n_sites, 3)) * np.nan

        # Similarly for coordination
        if solute_coordination:
            coord = job.ensure_namespace("coordination", scope=sol_results)
            if force_recalculate or not hasattr(coord, "number"):
                coord.number = np.ones(n_sites) * np.nan
                coord.closest = np.ones(n_sites) * np.nan
                coord.center, coord.smearing = center, fermi_smearing = \
                    cgb.coordination.auto_parameterize(job.par.xl.nn_dist, job.par.xl.snn_dist, fermi_smearing)

        # Lastly also for chem-strain decomposition:
        if decompose:
            proj_sol_data = getattr(job.project, "bulk_" + sol)
            if force_recalculate or not hasattr(proj_sol_data, "sol_strain"):
                job.project.par.output.decompose = True
                os.chdir(job.project.location)
                job.project.run_solute(sol, job.project.bulk.central_id, procs)
                os.chdir(sol_dir)
            if force_recalculate or not hasattr(sol_results, "mechanical_energy"):
                sol_results.mechanical_energy = np.ones(n_sites) * np.nan
                sol_results.chemical_energy = np.ones(n_sites) * np.nan

        # Calculate segregation etc. for each GB site
        for i, en in enumerate(sol_results.energy):

            if use_mod:
                if i % mod_m != mod_n:
                    continue

            if np.isnan(en):
                cgb.osio.tee(job.log_loc, "...For site " + str(i))
                # Go to the site's directory (creating it if needed)
                site_name = "site_" + str(i)
                cgb.osio.make_dir(site_name, allow_exists=True)
                os.chdir(site_name)

                # Calculate the segregation energy
                reference_structure_file = os.path.join(job.location, "gb.xyzin")
                species_string = " ".join([job.par.chem.host, sol])

                lammps = cgb.lammps.LammpsJob(cluster=(job.par.cluster.center, job.par.cluster.inner_r))
                lammps.add_structure_file(reference_structure_file)
                lammps.add_potential(job.par.chem.potential, species_string)
                lammps.add_species_change(i)
                lammps.add_dump_thermo(job.par.lammps.omega_step)
                if need_pos:
                    lammps.add_dump_xyz("last.xyz", job.par.lammps.omega_step)
                lammps.add_run_minimization(job.par.lammps.max_steps, job.par.lammps.force_convergence)
                lammps.run(job.procs)

                # Parse and save the energy
                seg_data = lammps.thermo_data
                bulk_data = job.project.bulk.thermo_data
                gb_data = job.gb.thermo_data
                sol_data = getattr(job.project, "bulk_" + sol).thermo_data
                e_bind = job.segregation_energy(seg_data, bulk_data, gb_data, sol_data)
                sol_results.energy[i] = e_bind

                # Calculate/save other results:
                if need_pos:
                    pos, spec = cgb.structure.read_xyz("last.xyz")

                    # Save structures if requested
                    if save_relaxed:
                        comment = "Segregation of " + sol + " to site " + str(i) + " for " + job.name
                        if save_radius is None:
                            cgb.structure.write_xyzin("gb_" + sol + "_" + str(i) + ".xyzin", pos, job.par.cluster.size,
                                                      species=spec, comment=comment)
                        else:  # To save on memory, you may wish to save only atoms near the GB site
                            comment += " in a radius of " + str(save_radius)
                            disp = pos - pos[i]
                            dist_sq = np.sum(disp * disp, axis=1)
                            mask = dist_sq < save_radius ** 2
                            cgb.structure.write_xyzin("gb_" + sol + "_" + str(i) + ".xyzin", pos, job.par.cluster.size,
                                                      species=spec, comment=comment, mask=mask)

                    # Save in-situ voro++ results on the solute site
                    if solute_voronoi:
                        vols, ars, els, verts, facs, eds, offs = cgb.voronoi.voronoi(pos, l, l, l)
                        voro.volume[i] = vols[i]
                        voro.area[i] = ars[i]
                        voro.edge_length[i] = els[i]
                        voro.vertices[i] = verts[i]
                        voro.faces[i] = facs[i]
                        voro.edges[i] = eds[i]
                        voro.offset[i] = offs[i]

                    # Save in-situ coordination data for the solute site
                    if solute_coordination:
                        number, closest = cgb.coordination.coordination(pos, coord.center, coord.smearing, indices=i)
                        coord.number[i] = number[0]  # coords is an array with one element, and we want a float
                        coord.closest[i] = closest[0]  # sim.

                    # Calculate the solute-relaxed structure sans solute
                    if decompose:
                        strain_structure_file = os.path.join(job.location, "monosegregation", sol, site_name,
                                                             "strained.xyzin")
                        cgb.structure.write_xyzin(strain_structure_file, pos, job.par.cluster.size, nspecies=1,
                                                  comment="Solute-relaxed structure with pure host chemistry.")

                        lammps = cgb.lammps.LammpsJob(cluster=(job.par.cluster.center, job.par.cluster.inner_r))
                        lammps.add_structure_file(strain_structure_file)
                        lammps.add_potential(job.par.chem.potential, job.par.chem.host)
                        lammps.add_dump_thermo(job.par.lammps.omega_step)
                        lammps.add_run_static()
                        lammps.run(job.procs)

                        seg_strain_data = lammps.thermo_data
                        bulk_strain_data = getattr(job.project, "bulk_" + sol).sol_strain.thermo_data

                        mech_e_bind = job.segregation_energy(seg_strain_data, bulk_data, gb_data, bulk_strain_data)
                        chem_e_bind = job.segregation_energy(seg_data, bulk_strain_data, seg_strain_data, sol_data)
                        sol_results.mechanical_energy[i] = mech_e_bind
                        sol_results.chemical_energy[i] = chem_e_bind

                        os.remove(strain_structure_file)

                    os.remove("last.xyz")

                job.save()

                os.chdir("..")
        os.chdir("..")
    os.chdir(os.path.join("..", ".."))

    end = time.time()
    cgb.osio.tee(job.log_loc, job.name + " monosegregation runtime = " + str(end - start) + "s.")


def main(args):
    start = time.time()

    cgb.osio.run_in_hierarchy(run_monosegregation, vars(args))

    end = time.time()
    print("Total monosegregation runtime = " + str(end - start) + "s.")


def _ret_parser():
    parser = argparse.ArgumentParser(description="Calculate the solute segregation energy of each solute for each GB "
                                                 "site at each GB recursively through the ClusterGB hierarchy.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--procs", "-np", type=int, default=1,
                        help="Max processors available to run mpi on.")
    parser.add_argument("--save-relaxed", "-save", action="store_true",
                        help="Save the relaxed structures.")
    parser.add_argument("--save-radius", "-rad", type=float, default=None,
                        help="Radius around the solute to save. (Default is entire structure.)")
    parser.add_argument("--force-recalculate", "-recalc", action="store_true",
                        help="Overwrite any existing segregation data with new calculations.")
    parser.add_argument("--solute-voronoi", "-voro", action="store_true",
                        help="Calculate the relaxed Voronoi properties of each site-segregated solute. Collected in "
                             "Job.results.voronoi.${SOLUTE}")
    parser.add_argument("--solute-coordination", "-coord", action="store_true",
                        help="Calculate the relaxed coordination number of each site-segregated solute. Collected in "
                             "Job.results.coord.${SOLUTE}")
    parser.add_argument("--fermi-smearing", "-smear", default=None,
                        help="Amount (angstroms) of fermi-smearing to apply when calculating coordination values. "
                             "(Default is 20 percent of the difference between first and second nearest neighbour "
                             "distances.)")
    parser.add_argument("--decompose", "-dcomp", action="store_true",
                        help="In addition to segregation calculations, run pure-host structures with solute-relaxed "
                             "structures so the segregation energy can be decomposed into chemical and deformation "
                             "components. (Activated by default if the project was initialized with this flag.)")
    parser.add_argument("--mod", "-mod", type=int, nargs=2, default=[None, None],
                        help="Only calculate monosegregation for sites whose site ids are n mod m. By giving values "
                             "0 through m-1 for the n-argument, one can paralellize the monosegregation calculations "
                             "over multipule calls. (**WARNING:** This isn't guaranteed to be stable and energies can"
                             "get missed if two calls are trying to update the same *Job* object at the same time.)")
    return parser


if __name__ == "__main__":
    returned_parser = _ret_parser()
    arguments = returned_parser.parse_args()
    main(arguments)
