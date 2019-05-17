#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
The `Job` class holds all the information about a particular GB--i.e. a name associated with a particular set of
macroscopic GB parameters. Parameters for how to treat this GB (relaxation etc.) are inherited from the Project object
in the directory where the Job is initialized. Results and analysis are stored in the `Job` namespace.
"""

import yaml
import os
import numpy as np
import clustergb as cgb
from argparse import Namespace
import pickle
from .project import Project
import shutil

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


class Job:
    """
    For calculations on a particular GB cluster.

    Attributes:
        job_save_name (str): File name for pickles of this object.
        project (Project): The `Project` this job belongs to.
        version (str): Version of code object was created with from config.yml.
        procs (int): How many processors to call LAMMPS with.
        name (str): Name of the Job.
        location (str): Path to where the Job is stored.
        par (Namespace): A collection of variables for running a calculation, most of which are pulled from the parent
          clustergb.project.Project object.

            - **output.thermo_period** (*int*) -- How many steps to go between writing LAMMPS thermodynamic data.
            - **output.dump_period** (*int*) -- How many steps to go between dumpy LAMMPS .xyz data.
            - **gb** (*Namespace*) -- A collection of GB macroscopic DoF parameters.

                - **misorientation** (*float*) -- Misorientation between the two grains.
                - **shared** (*np.ndarray*) -- :math:`(3,)` shared axis (in multiples of the bravais vectors) of the
                  two grains.
                - **normal** (*np.ndarray*) -- :math:`(3,)` grain boundary normal axis (in Bravais vectors) of one
                  grain.
                - **symmetric** (*bool*) -- Whether the boundary should be symmetric (opposite normals calculated using
                  misorientation).

        gb (Namespace): A space for storing results from the construction and microscopic optimization of the GB.

            - **annealing_quench_energies** (*np.ndarray*) -- Interfacial energies from each quench of the annealing
              process.
            - **brav1** (*np.ndarray*) -- :math:`(3,3)` Bravais lattice for the first grain (x-left).
            - **brav2** (*np.ndarray*) -- :math:`(3,3)` Bravais lattice for the second grain (x-right).
            - **gb_energy** (*float*) -- The interfacial energy (in mJ/m^2).
            - **natoms_nonbulklike** (*int*) -- The number of atoms inside `self.par.cluster.r_inner` with non-bulklike
              common neighbour analysis values.
            - **natoms_inner** (*int*) -- The number of bulklike atoms inside `self.par.cluster.r_inner`.
            - **natoms_outer** (*int*) -- The number of atoms beyond `self.par.cluster.r_inner` but inside
              `self.par.cluster.r_outer`.
            - **natoms_balance** (*int*) -- The remaining number of atoms in the cluster.
            - **spsa_trial_energies** (*np.ndarray*) -- Interfacial energies from each SPSA minimization trial.
            - **thermo_data** (*Namespace*) -- *LammpsJob.thermo_data* collection of thermodynamic properties for the
              final, relaxed grain boundary.

        results (Namespace): An empty container for other scripts to add results to. All `run` executables should save
          data to appropriately named sub-namespaces here.
    """

    job_save_name = "clustergb.job"

    def __init__(self, args):

        # TODO: Wrap the whole thing in a try clause that deletes the job from the parent before raising exception?
        project_file = Project.proj_save_name
        if not os.path.isfile(project_file):
            raise Exception("A " + self.__class__.__name__ + " must be initiated in the presence of a " + project_file)
        with open(project_file, 'rb') as f:
            self.project = pickle.load(f)

        # Record the version
        source_location = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(os.path.dirname(source_location), "config.yml"), "r") as f:
            config = yaml.load(f)
        self.version = config["version"]["version"]

        self.procs = None
        # We need the processor count as an attribute to keep the SPSA search running smoothly, but we'll keep it as
        # an explicit required argument for each external call that actually calculates something.
        self.debug = args.debug

        # Get the project settings, then add to them
        self.par = self.project.par
        self.par.gb = Namespace()

        # Make an empty container for calculated GB data
        self.gb = Namespace()

        if args.gb_style == "macro":  # Read from an [ijk](lmn)theta' format
            self.par.gb.misorientation = args.misorientation * np.pi / 180.
            self.par.gb.shared = np.array(args.shared)
            self.par.gb.normal = np.array(args.normal)
            self.par.gb.symmetric = False  # args.symmetric
            # Symmetry boundary functionality is not unique, so it is hidden from the user for now.
            # Calculate the required Bravais lattices to match GB macroscopic parameters
            brav1, brav2, new_shared = cgb.gb_geometry.find_bravais(self.par.gb.misorientation, self.par.gb.shared,
                                                                    self.par.gb.normal, self.par.gb.symmetric,
                                                                    self.par.xl.bravais, self.par.newton.precision,
                                                                    self.par.newton.verbosity,
                                                                    self.par.newton.max_steps)
        elif args.gb_style == "rot":  # Read a rotation matrix for each grain
            brav1 = self._rot_to_brav(np.array(args.rot1))
            brav2 = self._rot_to_brav(np.array(args.rot2))
        elif args.gb_style == "copy":  # Copy the Bravais lattices of another job
            with open(os.path.join(args.job_dir, self.__class__.job_save_name), 'rb') as f:
                image_job = pickle.load(f)
            self.par.gb = Namespace(**vars(image_job.par.gb))
            # Note: This isn't a true deep copy, but after the copied job has been saved and pickled it seems to be
            #       safely decoupled from the source Namespace.
            brav1 = np.array(image_job.gb.brav1)
            brav2 = np.array(image_job.gb.brav2)
        else:
            raise Exception("This portion of code should be inaccessible, please send a report to the author.")
        self.gb.brav1 = brav1
        self.gb.brav2 = brav2

        # Set output parameters
        # Note that these will overwrite copied values if you used an image Job
        self.par.output.thermo_period = args.thermo_period or self.par.lammps.omega_step
        self.par.output.dump_period = args.dump_period or self.par.lammps.omega_step

        # Make an empty container for other 'run' scripts to add results to
        self.results = Namespace()

        # Store the name
        self.name = args.name
        if self.name in self.project.child_names:
            raise Exception("Duplicate job name. Delete job using `cgb_delete.py`" + self.name + \
                            ", or use a different name.")
        self._add_to_parent()

        # Filestructure stuf
        self.job_save_name = self.__class__.job_save_name
        cgb.osio.make_dir(self.name)
        os.chdir(self.name)
        self.location = os.getcwd()

        # Initiate log file
        self.log_loc = os.path.join(self.location, self.job_save_name + ".log")
        cgb.osio.tee(self.log_loc, "Initializing a new " + self.__class__.__name__ + " in " + self.name)
        if args.gb_style == "macro":
            cgb.osio.tee(self.log_loc, "In the transformed coordinates, the shared axis is " + str(new_shared))

        self.save()

    def ensure_namespace(self, name, scope=None):
        """
        Ensure that a namespace exists, creating it if necessary.

        Args:
            name (str): The name of the Namespace.
            scope (object or Namespace): Where to look for it. (Default is self.)

        Returns:
            (*Namespace*) -- The namespace in question.
        """
        if scope is None:
            scope = self

        try:
            ns = getattr(scope, name)
        except AttributeError:
            setattr(scope, name, Namespace())
            ns = getattr(scope, name)
        self.save()
        return ns

    def new_boundary(self, procs):
        """
        Using the provided macroscopic GB parameters, make a new GB cluster, (stochasitcally) minimize its microscopic
        degrees of freedom, anneal it (if requested), then use common neighbour analysis to find GB sites. Write a new
        structure file (gb.xyzin) with this relaxed cluster having GB sites at the head of the file, followed by other
        atoms inside the inner radius, followed by atoms between inner and outer radii, followed by any other atoms.
        Saves data in the `self.gb` namespace.

        .. todo::

            Right now `Job` calls `structure.reorder` *after* minimization and annealing, but `Project` calls `reorder`
            on the ideal, unrelaxed bulk cluster. This doesn't make results wrong, but it is a weird inconsitency. Make
            it consistent.

        Args:
            procs (*int*): Number of processors to run LAMMPS on.
        """

        self.procs = procs

        # Minimize and anneal the boundary
        pos = self._minimize_boundary()
        pos, annealing_comment = self._anneal_boundary(pos)
        comment = self.par.cluster.comment + annealing_comment

        # Get CNA values for the minimized cluster
        cna_structure_file = "cna.xyzin"
        cgb.structure.write_xyzin(cna_structure_file, pos, self.par.cluster.size,
                                  comment="Minimized GB to be reordered")
        cna_file = "cna.dump"
        cgb.osio.tee(self.log_loc, "Performings common neighbour analysis...")
        cgb.lammps.run_cna(cna_structure_file, self.par.chem.potential, self.par.chem.host,
                           self.par.xl.type, self.par.xl.length, cna_file, nprocs=self.procs)
        cna_values = np.genfromtxt(cna_file, skip_header=9)
        os.remove(cna_structure_file)
        os.remove(cna_file)
        if self.debug:
            shutil.move("log.lammps", "log.cna.lammps")
        else:
            os.remove("log.lammps")

        # Reorder atoms according to the inner radius
        cgb.osio.tee(self.log_loc, "Reordering structure...")
        pos, counts = cgb.structure.reorder(pos, self.par.cluster.center, self.par.cluster.inner_r,
                                            self.par.cluster.outer_r,
                                            cna_vals=cna_values, lattice_type=self.par.xl.type)
        comment += cgb.structure.comment_count(counts)
        # Save the atom counts after reordering
        self.gb.natoms_nonbulklike = counts[0]
        self.gb.natoms_inner = counts[1]
        self.gb.natoms_outer = counts[2]
        self.gb.natoms_balance = counts[3]

        # Save the final gb energy
        gb_data = self.gb.thermo_data
        bulk_data = self.project.bulk.thermo_data
        gb_energy = self._grain_boundary_energy(gb_data, bulk_data, self.par.cluster.inner_r)
        self.gb.gb_energy = gb_energy

        # Save the updated results
        self.save()

        cgb.osio.tee(self.log_loc, "GB energy = " + str(gb_energy))

        # Write the final structure file
        cgb.structure.write_xyzin("gb.xyzin", pos, self.par.cluster.size, comment=comment, nspecies=2)

    def refresh_project(self):
        """
        Reloads the parent project object and saves it to `self.project` again, getting a fresh copy in case there
        have been changes (e.g. from the actions of another of the project's child jobs).
        """

        proj_loc = os.path.dirname(self.location)
        project_file = os.path.join(proj_loc, Project.proj_save_name)
        with open(project_file, 'rb') as f:
            self.project = pickle.load(f)
        self.save()

    def save(self):
        """Write a pickled version of this object."""
        with open(os.path.join(self.location, self.job_save_name), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def segregation_energy(gb_solute, bulk, gb, bulk_solute):
        """
        Calculate segregation energy for the solute to the GB site. Positive energy indicates favourable binding.
        Uses total energy in order to capture strain energy deposited by the solute outside r_inner. This is safe
        because the surface energy will almost perfectly cancel (unlike GB surface energy.)

        Each input is a namespace storing the thermodynamic data from a LAMMPS calculation, which will have an `etotal`
        value, storing the total energy from the calculation.

        Uses a convention with negative segregation energy as favourable.

        Args:
            gb_solute (Namespace): Data from a calculation with a solute atom at a GB site, e.g.
                                   `clustergb.lammps.LammpsJob.thermo_data`.
            bulk (Namespace): Data from a bulk calculation, e.g. `Project.bulk.thermo_data`.
            gb (Namespace): Data from an undecorated GB calculation, e.g. `Job.gb.thermo_data`.
            bulk_solute (Namespace): Data from a pure bulk calculation, e.g. `Project.bulk_X.thermo_data`.

        Returns:
            (*float*) -- Solute segregation energy.
        """

        return (gb_solute.etotal + bulk.etotal) - (gb.etotal + bulk_solute.etotal)

    def _add_to_parent(self):
        """At self to the parent's `child_names` list."""
        project_file = Project.proj_save_name
        with open(project_file, 'rb') as f:  # Get most up to date parent
            self.project = pickle.load(f)
        self.project.child_names += [self.name]
        self.project.save()

    def _anneal_boundary(self, pos):
        """
        Given a cluster defined by the atomic positions in `pos`, anneals the structure using a series of quench-and-
        minimize branches. The minimum GB energy branch is saved, but the total anneal process continues from the un-
        quenched state. i.e. we're splitting timelines and quenching after different annealing times and just saving
        the lowest GB energy structure we find across the multiverse.

        Args:
            pos (np.ndarray): :math:`(n,3)` positions of atoms in the annealing GB cluster.

        Returns:
            2-element tuple containing

            - (*np.ndarray*) -- :math:`(n,3)` annealed atomic positions.
            - (*str*) -- Header comment for the structure file to explain its history.
        """
        if self.par.lammps.annealing.time > 0:
            cgb.osio.tee(self.log_loc, "Annealing...")

            best_gb_energy = float(self.gb.gb_energy)  # Copy the minimization energy
            best_cycle = 0
            best_pos = pos
            best_data = Namespace(**vars(self.gb.thermo_data))  # Copy the minimization results
            self.gb.annealing_quench_energies = np.ones(self.par.lammps.quenching.samples) * np.nan

            restart_file_name = "restart.lammps"
            restart_file = None  # On the first cycle, we won't restart, later it will be set to `restart_file_name`
            structure_file_name = "anneal.xyzin"  # But rather we'll read the relaxed structure from merging
            structure_file = structure_file_name
            cgb.structure.write_xyzin(structure_file, pos, self.par.cluster.size)
            init_temp = self.par.lammps.annealing.init_temp

            for cycle in np.arange(1, self.par.lammps.quenching.samples + 1):
                # Anneal and quench structure, then minimize (write final minimized energy)
                lammps = cgb.lammps.LammpsJob(cluster=(self.par.cluster.center, self.par.cluster.inner_r))
                if restart_file is not None:
                    lammps.add_restart(restart_file)
                else:
                    lammps.add_structure_file(structure_file)
                lammps.add_potential(self.par.chem.potential, self.par.chem.host)
                lammps.add_run_nvt_langevin(self.par.lammps.annealing.steps, self.par.lammps.timestep,
                                            self.par.lammps.annealing.temp, self.par.lammps.annealing.langevin,
                                            init_temp=init_temp, write_restart=restart_file_name)
                lammps.add_run_nve_damped(self.par.lammps.quenching.steps, self.par.lammps.timestep,
                                          self.par.lammps.quenching.damping)
                lammps.add_dump_thermo(dump_period=self.par.output.thermo_period)
                lammps.add_dump_xyz("last.xyz", dump_period=self.par.output.dump_period)
                lammps.add_run_minimization(self.par.lammps.max_steps, self.par.lammps.force_convergence)
                lammps.run(self.procs)

                # For all but the 0th step, we'll be reading the lammps restart file to continue from where we left off
                restart_file = restart_file_name  # For subsequent cycles, read the restart file
                structure_file = None  # And thus we won't read the structure in anymore
                init_temp = None  # Similarly, we will be using the restart temperature

                # Read (annealed and quenched and) minimized GB energy
                gb_data = lammps.thermo_data
                bulk_data = self.project.bulk.thermo_data
                gb_energy = self._grain_boundary_energy(gb_data, bulk_data, self.par.cluster.inner_r)
                self.gb.annealing_quench_energies[cycle - 1] = gb_energy  # Cycles start counting at 1, but arrays at 0
                # Save the results if we've found a new minimum GB energy structure
                if gb_energy - best_gb_energy < 0:
                    best_gb_energy = gb_energy
                    best_pos, _ = cgb.structure.read_xyz("last.xyz")
                    best_cycle = cycle
                    best_data = Namespace(**vars(gb_data))  # Make a copy of the thermo data

                if self.debug:
                    shutil.move("log.lammps", "log.anneal_quench" + str(cycle) + ".lammps")

                cgb.osio.tee(self.log_loc, "Annealing cycle " + str(cycle) + ", quenched GB energy = " + str(gb_energy))

            # Now the GB is minimized with respect to merge distance and is annealed
            os.remove("last.xyz")
            os.remove(structure_file_name)
            os.remove(restart_file_name)
            time_used = (best_cycle / float(self.par.lammps.quenching.samples)) * self.par.lammps.annealing.time
            annealing_comment = "Annealing at " + str(self.par.lammps.annealing.temp) + " K for " + \
                                str(self.par.lammps.annealing.time) + " ps found the lowest energy after " + \
                                str(time_used) + " ps (" + str(best_cycle) + " cycles)"
            cgb.osio.tee(self.log_loc, annealing_comment)
            annealing_comment += "; "
            annealed_pos = best_pos
            self.gb.thermo_data = best_data
            self.gb.gb_energy = best_gb_energy
        else:
            annealing_comment = ""
            annealed_pos = pos

        self.save()
        return annealed_pos, annealing_comment

    def _gb_cost_function(self, u):
        """
        Given a microscopic DoF vector that includes `u[:3]` translation in reals space, and `u[3]` the merge cutoff,
        builds an array of atomic positions, statically calculates the cluster energy, and uses the unrelaxed bulk
        results from the parent Project to calculate and unrelaxed GB energy/area.

        Args:
            u (np.ndarray): :math:`(4,)` the translation and merge distance to use when constructing cluster
                            positions.

        Returns:
            (*float*) -- Energy of the unrelaxed GB surface.
        """

        # Decompose the vector we're optimizing on and bound the merge distance to something physically reasonable
        translation = u[:3]
        min_merge = self.par.spsa.merge_limits[0]
        max_merge = self.par.spsa.merge_limits[1]
        mid_merge = np.mean(self.par.spsa.merge_limits)
        merge_dist = cgb.extra_math.sigmoid(u[-1], left_asymptote=min_merge, right_asymptote=max_merge,
                                            center=mid_merge)

        # Create the two hemispheres of the boundary
        pos1, pos2 = cgb.gb_geometry.ideal_hemispheres(self.gb.brav1, self.gb.brav2, self.par.xl.basis, translation,
                                                       self.par.cluster.radius)
        merged_pos = cgb.gb_geometry.merge_hemispheres(pos1, pos2, merge_dist)
        merged_pos += self.par.cluster.center
        spsa_structure_file = "spsa.xyzin"
        cgb.structure.write_xyzin(spsa_structure_file, merged_pos, self.par.cluster.size)

        # Use LAMMPS to calculate the energy
        lammps = cgb.lammps.LammpsJob(cluster=(self.par.cluster.center, self.par.cluster.inner_r))
        lammps.add_structure_file(spsa_structure_file)
        lammps.add_potential(self.par.chem.potential, self.par.chem.host)
        lammps.add_dump_thermo(dump_period=1)
        lammps.add_run_static()
        lammps.run(self.procs)

        # Remove the tmp structure file
        os.remove(spsa_structure_file)

        # Calculate the unrelaxed GB energy
        gb_data = lammps.thermo_data
        bulk_data = self.project.bulk.thermo_data
        return self._grain_boundary_energy(gb_data, bulk_data, self.par.cluster.inner_r, unrelaxed=True)

    @staticmethod
    def _grain_boundary_energy(data_gb, data_bulk, r_inner, unrelaxed=False):
        """
        Returns the GB surface energy for the boundary. Uses only the energy of atoms within `r_inner` from the `center`
        point to avoid comparing clusters with different surface facets. For the purpose of calculating GB area, we
        assume that the GB plane is a flat disc based on the radius.

        Args:
            data_gb (Namespace): Data from an undecorated GB calculation, e.g. `Job.gb.thermo_data`.
            data_bulk (Namespace): Data from a bulk calculation, e.g. `Project.bulk.thermo_data`.
            r_inner (float): Radius inside which atoms for the results `thermo_data.c_pe_inner` energy come.
            unrelaxed (bool): Flag to use the initial (0th step) energies from the data Namespaces.

        Returns:
            (*float*) -- GB surface energy (mJ/m^2).
        """

        # TODO: Read r_inner directly from the results Namespace
        count_ratio = data_gb.v_n_inner / float(data_bulk.v_n_inner)
        eVangstromsq_to_mJmsq = 16021.8

        if unrelaxed:
            data_bulk = data_bulk.init_vals
            data_gb = data_gb.init_vals

        bulk_energy = data_bulk.c_pe_inner
        gb_energy = data_gb.c_pe_inner

        return eVangstromsq_to_mJmsq * (gb_energy - bulk_energy * count_ratio) / (np.pi * r_inner ** 2)

    def _minimize_boundary(self):
        """
        Once grain bravais lattices have been set to reflect macroscopic DoF, (stochasitically) minimizes microscopic
        DoF with respect to translation and merging atoms together across the GB plane. (The merging is not strictly
        necessary--translation is sufficient--but I found including the merge distance in the degrees of freedom helps
        to find a good solution faster.)

        Saves the best GB results found to `self.gb.thermo_data`.

        Returns:
            (*np.ndarray*) -- :math:`(n,3)` atomic positions in the SPSA-minimized GB cluster.
        """
        cgb.osio.tee(self.log_loc, "Minimizing microscopic DoF...")

        # Use SPSA to search the transition-merge space and look for the lowest GB energy
        # We do this several times and take the best result because SPSA is, as the name suggests, stochastic
        spsa = self.par.spsa  # Just a quick alias for less typing...
        spsa_structure_file = "spsa.xyzin"
        best_gb_energy = np.inf
        best_pos = None
        best_gb_data = None
        self.gb.spsa_trial_energies = np.ones(spsa.trials) * np.nan

        if (spsa.trials == 0) or (spsa.max_steps == 0):
            return self._no_search()

        for i in np.arange(spsa.trials):
            # Use SPSA to optimize the microscopic DoF from a (semi)random guess
            micro_dof, steps = cgb.extra_math.spsa(self._semirandom_dof(), self._gb_cost_function, spsa.probe_size,
                                                   spsa.step_size, spsa.conv_u, spsa.conv_J, max_steps=spsa.max_steps,
                                                   A=spsa.step_offset, alpha=spsa.alpha, gamma=spsa.gamma,
                                                   m=spsa.momentum, verbose=spsa.verbosity)

            # Use the optimized parameters to make a new structure
            translation = micro_dof[:3]
            min_merge = spsa.merge_limits[0]
            max_merge = spsa.merge_limits[1]
            mid_merge = np.mean(spsa.merge_limits)
            merge_dist = cgb.extra_math.sigmoid(micro_dof[-1], left_asymptote=min_merge, right_asymptote=max_merge,
                                                center=mid_merge)
            pos1, pos2 = cgb.gb_geometry.ideal_hemispheres(self.gb.brav1, self.gb.brav2, self.par.xl.basis, translation,
                                                           self.par.cluster.radius)
            merged_pos = cgb.gb_geometry.merge_hemispheres(pos1, pos2, merge_dist)
            merged_pos += self.par.cluster.center
            # TODO: Give merge_hemispheres a shift option so this can be one line everywhere

            cgb.structure.write_xyzin(spsa_structure_file, merged_pos, self.par.cluster.size)

            # TODO: This is inefficient; I'm just redoing the last calculation to get the lammps.thermo_data. Be better.
            lammps = cgb.lammps.LammpsJob(cluster=(self.par.cluster.center, self.par.cluster.inner_r))
            lammps.add_structure_file(spsa_structure_file)
            lammps.add_potential(self.par.chem.potential, self.par.chem.host)
            lammps.add_dump_thermo(dump_period=self.par.lammps.omega_step)
            lammps.add_dump_xyz("last.xyz", dump_period=self.par.lammps.omega_step)
            lammps.add_run_minimization(self.par.lammps.max_steps, self.par.lammps.force_convergence)
            lammps.run(self.procs)

            pos, _ = cgb.structure.read_xyz("last.xyz")
            gb_data = lammps.thermo_data
            bulk_data = self.project.bulk.thermo_data
            gb_energy = self._grain_boundary_energy(gb_data, bulk_data, self.par.cluster.inner_r)
            self.gb.spsa_trial_energies[i] = gb_energy
            cgb.osio.tee(self.log_loc, "Trial " + str(i) + " (in " + str(steps) + " steps), stochasitcally minimized "
                         "GB energy = " + str(gb_energy))
            if gb_energy < best_gb_energy:
                best_gb_energy = gb_energy
                best_pos = pos
                best_gb_data = Namespace(**vars(gb_data))  # Make a copy of the thermo data
                if self.debug:
                    shutil.move("log.lammps", "log.spsa_trial" + str(i) + "_step" + str(steps) + ".lammps")
        cgb.osio.tee(self.log_loc, "Best GB energy found = " + str(best_gb_energy))

        os.remove("last.xyz")
        os.remove(spsa_structure_file)

        self.gb.thermo_data = best_gb_data  # We'll overwrite this later after annealing (if there is any)
        self.gb.gb_energy = best_gb_energy
        self.save()
        return best_pos

    def _no_search(self):
        """
        A simplified alternative to SPSA for microscopic DoF minimization where we leave the grain translations at 0
        and just merge the minimum distance.

        Returns:
            (*np.ndarray*) -- :math:`(n, 3)` merged and minimized (force relaxation) positions.
        """
        # Without any explicit search, just use zero translation and merge the minimum distance
        pos1, pos2 = cgb.gb_geometry.ideal_hemispheres(self.gb.brav1, self.gb.brav2, self.par.xl.basis, np.zeros(3),
                                                       self.par.cluster.radius)
        merged_pos = cgb.gb_geometry.merge_hemispheres(pos1, pos2, self.par.spsa.merge_limits[0])
        merged_pos += self.par.cluster.center

        no_search_structure_file = "just_merge.xyzin"
        cgb.structure.write_xyzin(no_search_structure_file, merged_pos, self.par.cluster.size)

        lammps = cgb.lammps.LammpsJob(cluster=(self.par.cluster.center, self.par.cluster.inner_r))
        lammps.add_structure_file(no_search_structure_file)
        lammps.add_potential(self.par.chem.potential, self.par.chem.host)
        lammps.add_dump_thermo(dump_period=self.par.lammps.omega_step)
        lammps.add_dump_xyz("last.xyz", dump_period=self.par.lammps.omega_step)
        lammps.add_run_minimization(self.par.lammps.max_steps, self.par.lammps.force_convergence)
        lammps.run(self.procs)

        pos, _ = cgb.structure.read_xyz("last.xyz")
        gb_data = lammps.thermo_data
        bulk_data = self.project.bulk.thermo_data
        gb_energy = self._grain_boundary_energy(gb_data, bulk_data, self.par.cluster.inner_r)

        self.gb.thermo_data = gb_data  # We'll overwrite this later after annealing (if there is any)
        self.gb.gb_energy = gb_energy
        self.save()

        os.remove("last.xyz")
        os.remove(no_search_structure_file)
        return pos

    def _rot_to_brav(self, rot):
        """
        Transforms a rotation matrix and the initial Bravais lattice into a rotated bravais lattice, ala
        Olmsted-Foiles-Holm [1]_.

        .. [1] Olmsted, Foiles, Holm, Acta Mater. 57 (2009)

        .. warning::

            This is working fine for cubic crystals, but if you implement hcp (or similar) and get weird behaviour,
            this is a spot to examine.

        Args:
            rot (np.ndarray or list): :math:`(9,)` The rotation matrix row by row.

        Returns:
            (*np.ndarray*) -- :math:`(3,3)` the corresponding Bravais lattice.
        """
        rot = np.reshape(rot, (3, 3))
        # Renormalize by row
        for n in np.arange(rot.shape[0]):
            rot[n, :] = cgb.extra_math.l2normalize(rot[n, :])
        # Double check that it's a rotation matrix
        cgb.extra_math.is_rotation_matrix_3d(rot)
        # The give info in terms of XL and want sample and we do the opposite (or vice versa)...
        # The point is, we want the exact opposite rotation
        ofh_transp = np.transpose(rot)
        # Rescale to lattice units and return
        return ofh_transp * self.par.xl.length

    def _semirandom_dof(self):
        """
        Generate a random set of translations (modulo the second grain's bravais lattice) and merge distance (inside
        the merge limits) to use as an initial guess for SPSA minimization of the microscopic DoF.

        Returns:
            (*np.ndarray*) -- Parameter vector for microscopic DoF (three translations and a merge distance).
        """
        trans = np.dot(self.gb.brav2, 0.5 * np.random.rand(3))
        # We know the biggest possible needed jump will be lattice / 2
        merge = np.random.uniform(self.par.spsa.merge_limits[0], high=self.par.spsa.merge_limits[1])
        return np.append(trans, merge)
