#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Projects control the workflow to ensure that multiple grain boundaries are treated with the same set of operations,
e.g. annealing temperatures and times.

.. note::

    Project objects save themselves using pickle, but they can only be unpickled IF clustergb is added to the
    Python path, e.g. `import sys; sys.path.append('/path/to/clustergb')`.
"""

import yaml
import os
import numpy as np
import clustergb as cgb
from argparse import Namespace
import pickle

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


class Project:
    """
    ClusterGB rests on LAMMPS, which uses classical potential to evaluate forces and energies.

    Projects store parameters which will be used across many individual grain boundaries (i.e. Jobs). This ensures
    consistency in how these boundaries are treated. Each Project is associated with a particular potential file and
    chemical symbols for the host and solute species.

    After initialization, Projects have methods for performing a calculation of a cluster of perfect bulk, and bulk
    with a substitutional defect at the center of the cluster.

    All units are assumed to be LAMMPS "metal" units (angstroms, eV, etc...)

    .. todo::

        Expand Attributes description of what is stored in bulk and bulk_${SOLUTE} Namespaces.

    Attributes:
        proj_save_name (str): File name for pickles of this object.
        version (str): Version of code object was created with (from config.yml).
        name (str): Name of the Project.
        child_names (list): A list of the names of Job objects lower in the hierarchy.
        location (str): Path to where the Project is stored.
        par (Namespace): A collection of parameters for running cluster calculations.

            - **chem** (*Namespace*) -- A collection of parameters for the chemistry.

                - **host** (*str*) -- Chemical symbol for host species.
                - **potential** (*str*) -- Path to the LAMMPS potential to use.
                - **solutes** (*list*) -- List of chemical symbols for the solute species (if any).

            - **xl** (*Namespace*) -- A collection of parameters for the crystal structure.

                - **bravais** (*np.ndarray*) -- :math:`(3,3)` Bravais vectors for the crystal lattice unit cell.
                - **basis** (*np.ndarray*) -- :math:`(3,3)` direct-coordinate basis for atoms in the unit cell.
                - **type** (*str*) -- Short-hand for the crystal lattice name.
                - **length** (*float*) -- Length of the lattice vector. (Currently we use only cubic lattices, which
                  have just a single length scale.)
                - **nn_dist** (*float*) -- First-nearest neighbour distance for the lattice.
                - **snn_dist** (*float*) -- Second-nearest neighbour distance for the lattice.

            - **cluster** (*Namespace*) -- A collection of variables for cluster geometry.

                - **radius** (*float*) -- How big the cluster is.
                - **inner_r** (*float*) -- Radius of the sub-sphere used for calculating GB energies and finding
                  GB sites.
                - **outer_r** (*float*) -- A slightly larger sub-sphere, for moving a few extra atoms to the
                  head structure file. (Not yet used for anything.)
                - **vacuum** (*float*) -- The minimum vacuum between periodic images of the cluster.
                - **size** (*float*) -- Edge length of the cubic simulation box.
                - **center** (*np.ndarray*) -- :math:`(3,)` coordinates for the center of the simulation box.
                - **comment** (*str*) -- Header comment for structure files describing the cluster geometry.

            - **newton** (*Namespace*) -- A collection of parameters for running Newton's method to find grains'
              Bravais.

                - **precision** (*float*) -- Numeric precision limit, mostly for L2-norm difference determining
                  whether two vectors are the same. Used in gradient descent when looking for GB bravais vectors.
                - **max_steps** (*int*) -- Maximum number of gradient descent steps to use in search for rotation
                  angles when looking for GB bravais vectors.
                - **verbosity** (*int*) -- Verbosity interval of the gradient descent search for rotation angles.

            - **spsa** (*Namespace*) -- A collection of parameters for running SPSA to minimize microscopic DoF.

                - **probe_size** (*float*) -- How far to step when evaluating the simultaneous perturbation for a
                  gradient.
                - **gamma** (*float*) -- The power at which probe size decays with iterations.
                - **step_size** (*float*) -- Multiplier for how far to follow the gradient.
                - **step_offset** (*float*) -- Iteration offset for damping the decay of the step size.
                - **alpha** (*float*) -- The power at which the step size decays with iterations.
                - **conv_u** (*float*) -- The maximum change in (translation, merge)-space between consecutive minima
                  to be considered converged.
                - **conv_J** (*float*) -- The maximum change in unrelaxed GB energy between consecutive minima to be
                  considered converged.
                - **momentum** (*float*) -- Momenta for the stochastic gradient descent, i.e. fraction of last
                  iteration's step to add to the current step between 0 and 1.
                - **max_steps** (*int*) -- Maximum number of iterations to use.
                - **trials** (*int*) -- Number of (semi)randomly initialized SPSA trials to perform.
                - **verbosity** (*int*) -- Flag to activate verbose output for the SPSA search.
                - **merge_limits** (*np.ndarray*) -- :math:`(2,)` Max and min merge distances used in minimization of
                  microscopic DoF.

            - **lammps** (*Namespace*) -- A collection of parameters for LAMMPS runs.

                - **force_convergence** (*float*) -- Force convergence for all LAMMPS minimizations.
                - **max_steps** (*int*) -- Max steps for LAMMPS energy minimization.
                - **timestep** (*float*) -- Timestep for LAMMPS MD.
                - **omega_step** (*int*) -- The most steps any LAMMPS run will ever have.
                - **annealing** (*Namespace*) -- A collection of parameters for annealing.

                    - **time** (*float*) -- How long to anneal for.
                    - **temp** (*float*) -- How hot to anneal at.
                    - **init_temp** (*float*) -- Temperature for initializing velocities.
                    - **steps** (*int*) -- Number of LAMMPS steps to anneal for.
                    - **langevin** (*float*) -- Langevin damping parameter to use for thermostat while annealing
                      (see LAMMPS documentation_ for details.)

                - **quenching** (*Namespace*) -- A collection of parameters for quenching.

                    - **samples** (*int*) -- Number of times to sample the annealed state with a quench and
                      minimization.
                    - **damping** (*float*) -- Strength of quench (friction: force/velocity).
                    - **time** (*float*) -- Duration of quench.
                    - **steps** (*int*) -- Number of LAMMPS steps for the quench.

            - **output** (*Namespace*) -- Variables for what to output/calculate.

                - **decompose** (*bool*) -- Flag to run pure-host energy evaluations on solute-relaxed structures.

        bulk (Namespace): A space for collecting data from a run of the pure bulk structure.

            - **central_id** (*int*) -- ID of the center-most atom (don't for get to add "1" when looking for it in a
              LAMMPS file, since LAMMPS starts counting at "1" while Python starts at "0").
            - **thermo_data** (*Namespace*) -- Thermodynamic data for the bulk cluster run parsed from the LAMMPS log.

        bulk_${SOLUTE} (Namespace): Space(s) for collecting data from runs of the bulk-like structure with a single
          ${SOLUTE} species substitutional defect.

            - **thermo_data** (*Namespace*) -- Thermodynamic data for the parsed from the LAMMPS log for the calculation
              with a single ${SOLUTE} atom in the center of the cluster.


    .. _documentation: http://lammps.sandia.gov/doc/fix_langevin.html
    """

    proj_save_name = "clustergb.project"

    def __init__(self, args):
        """
        Parse the command line arguments that are passed in to initialize a project.

        .. todo::

            Get __init__ docs showing in the html!

        Args:
            args (Namespace): Including

                - **name** (*str*) -- A name for the project.
                - **potential** (*str*) -- Path to the LAMMPS potential to use.
                - **fcc**, **bcc**, xor **hcp** (*float*) -- Which underlying crystal structure to use.
                - **args.solutes** (*list*) -- List of chemical symbols for the solute species (if any).
                - **radius**, **inner_radius**, and **outer_radius** (*float*) -- (Sub)cluster radii.
                - **vacuum** (*float*) -- Amount of vacuum between periodic cluster images.
                - **newton_precision** (*float*) -- Precision when searching for rotated Bravais lattices.
                - **newton_max_steps** and **newton_verbosity** (*int*) -- Max steps and verbosity for Newton search.
                - **spsa_probe_size**, **_gamma**, **_step_size**, **_step_offset**, **_alpha**,
                  **_convergence_distance**, **_convergence_energy**, and **_momentum** (*float*) --
                  Numerical parameters for SPSA search of best microscopic GB DoF.
                - **spsa_max_steps**, **_trials** (*int*) -- The maximum number of steps in a single SPSA search and how
                                                             many differently seeded searches to attempt, respectively.
                - **spsa_verbose** (*bool*) -- Whether to print SPSA output.
                - **merge_limits** (*tuple*) -- Minimum and maximum *float* limits for the merge distance in the SPSA
                                                search.
                - **lammps_force_convergence** (*float*) -- Force convergence limit for LAMMPS minimizations.
                - **lammps_max_steps** (*int*) -- Max steps for LAMMPS minimizations.
                - **lammps_timestep** (*float*) -- Timestep for LAMMPS MD runs.
                - **annealing_time**, **_temp**, **_init_temp** (*float*) -- Total time and (initial) temperature for GB
                                                                             annealing.
                - **quenching_samples** (*int*) -- How many times to quench the anneal searching for the best energy.
                - **quenching_damping** and **_time** (*float*) -- Damping parameter to suck out energy during a quench
                                                                   and how long to do it for.
                - **decompose** (*bool*) -- Whether to run the extra calculations necessary to decompose energies into
                                            "mechanical" and "chemical" components.
        """

        # Make sure you"re not already in a disallowed portion of the project hierarchy
        for fname in ["clustergb.project", "clustergb.job"]:
            if os.path.isfile(fname):
                raise Exception("Cannot initiate " + self.__class__.__name__ + " where there is a " + fname + " file.")

        # Record the version
        source_location = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(os.path.dirname(source_location), "config.yml"), "r") as f:
            config = yaml.load(f)
        self.version = config["version"]["version"]

        # Store the name
        self.name = args.name
        self.child_names = []

        # Get the parameters for GB calculations
        self.par = Namespace()
        # Set the material
        self.par.chem = Namespace()
        self.par.chem.host = args.host
        self.par.chem.potential = os.path.abspath(args.potential)
        self.par.chem.solutes = args.solutes

        # Make a new directory for this object and go there
        cgb.osio.make_dir(self.name)
        os.chdir(self.name)
        self.location = os.getcwd()

        # Initiate log file
        self.log_loc = os.path.join(self.location, self.proj_save_name + ".log")
        cgb.osio.tee(self.log_loc, "Initializing a new " + self.__class__.__name__ + " in " + self.name)

        # Set crystal parameters
        self.par.xl = Namespace()
        if args.fcc:
            self.par.xl.bravais = np.matrix([[1., 0., 0.],
                                             [0., 1., 0.],
                                             [0., 0., 1.]]) * args.fcc
            self.par.xl.basis = np.matrix([[0.0, 0.0, 0.0],
                                           [0.5, 0.5, 0.0],
                                           [0.5, 0.0, 0.5],
                                           [0.0, 0.5, 0.5]])  # Internal coordinates
            self.par.xl.type = "fcc"
            self.par.xl.length = args.fcc
            self.par.xl.nn_dist = args.fcc * np.sqrt(2 * 0.25)  # Distance to middle of surface of cubic cell
            self.par.xl.snn_dist = args.fcc  # Second nearest neighbour
        elif args.bcc:
            self.par.xl.bravais = np.matrix([[1., 0., 0.],
                                             [0., 1., 0.],
                                             [0., 0., 1.]]) * args.bcc
            self.par.xl.basis = np.matrix([[0.0, 0.0, 0.0],
                                           [0.5, 0.5, 0.5]])  # Internal coordinates
            self.par.xl.type = "bcc"
            self.par.xl.length = args.bcc
            self.par.xl.nn_dist = args.bcc * np.sqrt(3 * 0.25)  # Distance to center of cubic cell
            self.par.xl.snn_dist = args.bcc  # Second nearest neighbour
        elif args.hcp:
            raise NotImplementedError

        # Set cluster geometry
        self.par.cluster = Namespace()
        self.par.cluster.radius = args.radius
        self.par.cluster.inner_r = args.inner_radius
        self.par.cluster.outer_r = args.outer_radius
        if self.par.cluster.outer_r is None:
            self.par.cluster.outer_r = min(self.par.cluster.inner_r + 5., self.par.cluster.radius)
        if self.par.cluster.outer_r > self.par.cluster.radius:
            raise ValueError("The outer radius must be <= the cluster radius.")
        if self.par.cluster.inner_r > self.par.cluster.outer_r:
            raise ValueError("The inner radius must be <= the outer radius.")
        self.par.cluster.vacuum = args.vacuum
        self.par.cluster.size = 2 * args.radius + args.vacuum
        self.par.cluster.center = 0.5 * self.par.cluster.size * np.ones(3)
        self.par.cluster.comment = cgb.structure.comment_radii(self.par.cluster.inner_r, self.par.cluster.outer_r,
                                                               self.par.cluster.radius)

        # Set numerics for finding the GB rotated bravais vectors
        self.par.newton = Namespace()
        self.par.newton.precision = args.newton_precision
        self.par.newton.max_steps = args.newton_max_steps
        self.par.newton.verbosity = args.newton_verbosity

        # Set parameters for using SPSA to minimize the static GB energy
        self.par.spsa = Namespace()
        self.par.spsa.probe_size = args.spsa_probe_size
        if self.par.spsa.probe_size is None:
            # Default the SPSA probe (how far to perturb when calculating the gradient) to 1% of the lattice vector
            self.par.spsa.probe_size = 0.01 * self.par.xl.length
        self.par.spsa.gamma = args.spsa_gamma
        self.par.spsa.step_size = args.spsa_step_size
        self.par.spsa.step_offset = args.spsa_step_offset
        self.par.spsa.alpha = args.spsa_alpha
        self.par.spsa.conv_u = args.spsa_convergence_distance
        self.par.spsa.conv_J = args.spsa_convergence_energy
        self.par.spsa.momentum = args.spsa_momentum
        self.par.spsa.max_steps = args.spsa_max_steps
        self.par.spsa.trials = args.spsa_trials
        self.par.spsa.verbosity = args.spsa_verbose

        # Convert the minimum, maximum, and steps over which to scan for merge distances into an array of distances to
        # use. By default, scan from half the nearest-neighbour (NN) distance up to all of it
        if args.merge_limits is None:
            minimum = (1. / 3.) * self.par.xl.nn_dist
            maximum = 0.85 * self.par.xl.nn_dist
            # Defaults match Olmsted, Foiles, and Holm, Acta Mat 57 (2009)
        else:
            minimum = float(args.merge_limits[0])
            maximum = float(args.merge_limits[1])
        self.par.spsa.merge_limits = np.array([minimum, maximum])

        # Set LAMMPS parameters
        self.par.lammps = Namespace()
        self.par.lammps.force_convergence = args.lammps_force_convergence
        self.par.lammps.max_steps = args.lammps_max_steps
        self.par.lammps.timestep = args.lammps_timestep

        # Also for annealing
        self.par.lammps.annealing = Namespace()
        self.par.lammps.annealing.time = args.annealing_time
        self.par.lammps.annealing.temp = args.annealing_temp
        # (Ideally, to help the thermostat we would use 2*T since about half of the initial KE will get eaten up as PE,
        #  however, this might bring us too close to/above the melting point, so be a bit more conservative and just use
        #  1.5x instead)
        self.par.lammps.annealing.init_temp = args.annealing_init_temp
        if self.par.lammps.annealing.init_temp is None:
            self.par.lammps.annealing.init_temp = 1.5 * self.par.lammps.annealing.temp
        # Calculate the number of steps to use while annealing and quenching
        try:
            annealing_period = args.annealing_time / float(args.quenching_samples)
        except ZeroDivisionError:
            annealing_period = 0.
        self.par.lammps.annealing.steps = int(annealing_period / args.lammps_timestep)
        self.par.lammps.annealing.langevin = args.annealing_langevin

        # Also for quenching
        self.par.lammps.quenching = Namespace()
        self.par.lammps.quenching.samples = args.quenching_samples
        self.par.lammps.quenching.damping = args.quenching_damping
        self.par.lammps.quenching.time = args.quenching_time
        self.par.lammps.quenching.steps = int(args.quenching_time / args.lammps_timestep)

        # It will often be useful (for dumping) to know the largest possible step we'll ever see
        self.par.lammps.omega_step = max(self.par.lammps.max_steps, self.par.lammps.annealing.steps,
                                         self.par.lammps.quenching.steps)

        self.par.output = Namespace()
        self.par.output.decompose = args.decompose

        # Make empty containers for holding calculation data
        self.bulk = Namespace()
        for sol in self.par.chem.solutes:
            setattr(self, "bulk_" + sol, Namespace())

        self.save()

    def find_children(self):
        """
        Look in all lower directories for Job objects whose parent has the same name as self and ensure that these Jobs
        are listed in `self.child_names`.
        """

        dirs = [x[0] for x in os.walk(self.location)][1:]  # First element is self.location, the rest are child dirs
        for dir_ in dirs:
            try:
                job = cgb.osio.load_job(os.path.join(self.location, dir_))
            except IOError:
                continue

            if (job.project.name == self.name) and (job.name not in self.child_names):
                print "Adding from " + dir_
                self.child_names += [job.name]

    def run_bulk(self, procs):
        """
        Runs a LAMMPS minimization of the pure bulk-like cluster and saves thermodynamic output to the Namespace
        `self.bulk`. The id of the center-most atom is also stored in `self.bulk.central_id`.

        .. todo::

            Right now `Job` calls `structure.reorder` *after* minimization and annealing, but `Project` calls `reorder`
            on the ideal, unrelaxed bulk cluster. This doesn't make results wrong, but it is a weird inconsitency. Make
            it consistent.


        Returns:
            (*int*) -- The index of the atom closest to the center of the cluster.
        """

        cgb.osio.tee(self.log_loc, "Running bulk cluster")

        # Generate positions for a bulk cluster
        xaxis = np.array([1, 0, 0])
        zaxis = np.array([0, 0, 1])
        merge_overlapping = 0.1  # Just to catch in case we wind up with overlapping atoms
        brav1, brav2, _ = cgb.gb_geometry.find_bravais(0., zaxis, xaxis, False, self.par.xl.bravais,
                                                       self.par.newton.precision, False, self.par.newton.max_steps)
        pos1, pos2 = cgb.gb_geometry.ideal_hemispheres(brav1, brav2, self.par.xl.basis, np.zeros(3),
                                                       self.par.cluster.radius)
        pos = cgb.gb_geometry.merge_hemispheres(pos1, pos2, merge_overlapping)
        pos += self.par.cluster.center
        pos, counts = cgb.structure.reorder(pos, self.par.cluster.center,
                                            self.par.cluster.inner_r, self.par.cluster.outer_r)

        # Find the atom closest to the center
        dist_sq = np.sum((pos - self.par.cluster.center) ** 2, 1)
        central_id = np.argmin(dist_sq)

        # Write structure file
        comment = self.par.cluster.comment + cgb.structure.comment_count(counts)
        cgb.structure.write_xyzin("bulk.xyzin", pos, self.par.cluster.size, comment=comment)

        # Make a subdirectory for this calculation and go there
        cgb.osio.make_dir("bulk")
        os.chdir("bulk")

        # Run LAMMPS minimization of pure bulk cluster
        lammps = cgb.lammps.LammpsJob(cluster=(self.par.cluster.center, self.par.cluster.inner_r))
        lammps.add_structure_file(os.path.join(self.location, "bulk.xyzin"))
        lammps.add_potential(self.par.chem.potential, self.par.chem.host)
        lammps.add_dump_thermo(dump_period=self.par.lammps.omega_step)
        lammps.add_dump_xyz('last.xyz', dump_period=self.par.lammps.omega_step)
        lammps.add_run_minimization(self.par.lammps.max_steps, self.par.lammps.force_convergence)
        lammps.run(procs)

        # Save the data
        bulk_thermo_data = lammps.thermo_data
        self.bulk.thermo_data = bulk_thermo_data
        self.bulk.central_id = central_id
        self.save()

        # And write the relaxed bulk structure
        pos, _ = cgb.structure.read_xyz("last.xyz")
        os.remove("last.xyz")
        cgb.structure.write_xyzin(os.path.join(self.location, "bulk.xyzin"), pos, self.par.cluster.size,
                                  nspecies=2, comment=comment)
        # Note: The nspecies=2 is so we can later switch one of the atoms to be a solute

        # Get out
        os.chdir(self.location)

        return central_id

    def run_in_children(self, func, kwargs):
        """
        Run a function across all child Jobs.

        Args:
            func (callable): The function to call, which takes as its arguments a `job.Job` and any keyword arguments.
            kwargs (dict): The keyword arguments for `func`.
        """
        for job_name in self.child_names:
            job = cgb.osio.load_job(path=job_name)
            func(job, **kwargs)

    def run_solute(self, solute, solute_ids, procs):
        """
        Runs a LAMMPS minimization of the bulk-like cluster with a solute atoms replacing all the host sites indicated
        by `solute_ids`, and saves thermodynamic output to the Namespace `self.results.bulk_${solute}`.

        If the `par.output.decompose` flag is thrown, a second calculation is made for the solute-relaxed structure
        with pure-host chemsitry and saved to `self.results.bulk_${solute}.sol_strain`.

        Args:
            solute (str): Chemical symbol for the solute species.
            solute_ids (int or list): Which indices to convert to the solute species.
            procs (int): How many processors to run LAMMPS on.
        """

        # Get in
        name = "bulk_" + solute
        cgb.osio.make_dir(name, allow_exists=True)
        os.chdir(name)

        cgb.osio.tee(self.log_loc, "Running bulk-like cluster with a substitutional " + solute + " defect.")

        # Run LAMMPS minimization of bulk cluster with a solute
        lammps = cgb.lammps.LammpsJob(cluster=(self.par.cluster.center, self.par.cluster.inner_r))
        lammps.add_structure_file(os.path.join(self.location, "bulk.xyzin"))
        lammps.add_potential(self.par.chem.potential, self.par.chem.host + ' ' + solute)
        lammps.add_species_change(solute_ids)
        lammps.add_dump_thermo(dump_period=self.par.lammps.omega_step)
        lammps.add_dump_xyz("last.xyz", dump_period=self.par.lammps.omega_step)
        lammps.add_run_minimization(self.par.lammps.max_steps, self.par.lammps.force_convergence)
        lammps.run(procs)

        # Save the data
        sol_thermo_data = lammps.thermo_data
        sol_data = getattr(self, name)
        sol_data.thermo_data = sol_thermo_data
        self.save()

        if self.par.output.decompose:
            pos, _ = cgb.structure.read_xyz("last.xyz")
            sol_rel_file = os.path.join(self.location, name + ".xyzin")
            cgb.structure.write_xyzin(sol_rel_file, pos, self.par.cluster.size,
                                      nspecies=1, comment="Solute-relaxed structure with pure host chemistry.")

            # Run a static snapshot of the solute-relaxed structure with pure-host chemistry
            lammps = cgb.lammps.LammpsJob(cluster=(self.par.cluster.center, self.par.cluster.inner_r))
            lammps.add_structure_file(sol_rel_file)
            lammps.add_potential(self.par.chem.potential, self.par.chem.host)
            lammps.add_dump_thermo(dump_period=self.par.lammps.omega_step)
            lammps.add_run_static()
            lammps.run(procs)

            # Save the data
            strain_thermo_data = lammps.thermo_data
            sol_data.sol_strain = Namespace()
            sol_data.sol_strain.thermo_data = strain_thermo_data
            self.save()
            os.remove(sol_rel_file)

        os.remove("last.xyz")

        # Get out
        os.chdir(self.location)

    def save(self):
        """Save the Project as a pickled object."""
        with open(os.path.join(self.location, self.proj_save_name), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
