#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
A class and functions for writing LAMMPS input files and running LAMMPS.
"""

import os
import clustergb as cgb
from . import osio
import subprocess as sp
import argparse
import numpy as np
import yaml

__author__ = "Liam Huber, Raheleh Hadian"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def energy_volume_curve(xl_type, lattice_constants, pot_file, species, cell_repetitions=1,
                        solute_ids=None, new_type=2, input_file='in.rescale.lammps', nprocs=1):
    """
    Run a series of static calculations with different volumetric strains.

    Args:
        xl_type (str): Crystal structure identifier.
        lattice_constants (list): *float* lattice parameters to use.
        pot_file (str): Path to empirical potential to use.
        species (str): Space separated, potential-appropriate chemical symbols who will be applied (in order) to
                       the type column of the input file.
        cell_repetitions (int): How many repetitions of the unit cell to allow (in each of x-, y-, and
                                z-directions.)
        solute_ids (int or np.ndarray or list): Integer id(s) for which to change species. (Default is None.)
        new_type (int): New species value for solute ids. (Default is 2.)
        input_file (str): File to write LAMMPS input script to. (Default is "in.rescale.lammps")
        nprocs (int): How many processors to run on. (Default is 1.)

    Returns:
        2-element tuple containing

        - (*np.ndarray*) -- Supercell volumes used for strained structure.
        - (*np.ndarray*) -- Resulting total energies from a static calculation.
    """

    volumes = np.empty(len(lattice_constants))
    energies = np.empty(len(lattice_constants))
    for i, latt in enumerate(lattice_constants):
        data = run_static_bulk(xl_type, latt, pot_file, species, cell_repetitions=cell_repetitions,
                               solute_ids=solute_ids, new_type=new_type, input_file=input_file, nprocs=nprocs)
        volumes[i] = data.lx * data.ly * data.lz
        energies[i] = data.etotal
    return volumes, energies


def potential_check(pot_file):
    """
    Checks if the provided string is a path pointing to a valid file with a known extension for empirical potentials.

    .. warning:

        Doesn't actually check that it's a healthy potential file, just that it exists and is dressed up like one.

    Args:
        pot_file (str): Path to empirical potential to verify.

    Returns:
        (*str*) -- Path to empirical potential.

    Raises:
        Exception: When the path isn't to a file ending with the extension ".fs" or ".alloy"
    """

    extension = osio.file_extension(pot_file)
    valid_formats = ['fs', 'alloy']
    if extension not in valid_formats:
        raise Exception('Invalid potential format, please choose a file with a format among ' + str(valid_formats))
    return pot_file


def run_cna(structure, pot_file, species, xl_type, lattice_constant, cna_file, input_file='in.cna.lammps', nprocs=1):
    """
    Calculate the common neighbour analysis values for a structure.

    Args:
        structure:
        pot_file (str): Path to empirical potential to use.
        species (str): Space separated, potential-appropriate chemical symbols who will be applied (in order) to the
                       type column of the input file.
        xl_type (str): Crystal structure indicator.
        lattice_constant (float): Lattice parameter.
        cna_file (str): Where to dump the per-atom CNA data.
        input_file (str): File to write LAMMPS input script to. (Default is "in.cna.lammps")
        nprocs (int): How many processors to run on. (Default is 1.)
    """

    lammps = LammpsJob(input_file=input_file)
    lammps.add_structure_file(structure)
    lammps.add_potential(pot_file, species)
    lammps.add_dump_cna(xl_type, lattice_constant, cna_file)
    lammps.add_run_static()
    lammps.run(nprocs)
    return


def run_minimization_bulk(xl_type, lattice_constant, pot_file, species, cell_repetitions=1,
                          solute_ids=None, new_type=2, input_file='in.clustergb.lammps', nprocs=1,
                          pressure=0, isotropy='iso', max_steps=1000, force_convergence=0.0001,):
    """
    Optimize a structure to minimize forces.

    Args:
        xl_type (str): Crystal structure indicator.
        lattice_constant (float): Lattice parameter.
        pot_file (str): Path to empirical potential to use.
        species (str): Space separated, potential-appropriate chemical symbols who will be applied (in order) to the
                       type column of the input file.
        cell_repetitions (int): How many repetitions of the unit cell to allow (in each of x-, y-, and
                                z-directions.)
        solute_ids (int or np.ndarray or list): Integer id(s) for which to change species. (Default is None.)
        new_type (int): New species value for solute ids. (Default is 2.)
        input_file (str): File to write LAMMPS input script to. (Default is "in.clustergb.lammps")
        nprocs (int): How many processors to run on. (Default is 1.)
        pressure (float): Pressure at which to run in bars (default is 0.)
        isotropy (str): Allow box to relax 'iso'tropically or 'aniso'tropically (orthogonal only.) (Default is iso.)
        max_steps (int): Maximum number of CG steps to take. (Default is 1000)
        force_convergence (float): Stopping threshold based on the L2 norm of the global force vector. (Default is
                                   0.0001.)

    Returns:
        (*Namespace*) -- Parsed thermodynamics data from the log file.
    """
    lammps = cgb.lammps.LammpsJob(input_file=input_file)
    lammps.add_structure_bulk(xl_type, lattice_constant, cell_repetitions, nspec=len(species.split()))
    lammps.add_potential(pot_file, species)
    if solute_ids is not None:
        lammps.add_species_change(solute_ids, new_type=new_type)
    lammps.add_cell_relax(pressure=pressure, isotropy=isotropy)
    lammps.add_dump_thermo()
    lammps.add_run_minimization(max_steps, force_convergence)
    lammps.run(nprocs)
    return lammps.thermo_data


def run_static_bulk(xl_type, lattice_constant, pot_file, species, cell_repetitions=1,
                    solute_ids=None, new_type=2, input_file='in.clustergb.lammps', nprocs=1):
    """
    Run a static calculation of a bulk unit cell.

    Args:
        xl_type (str): Crystal structure indicator.
        lattice_constant (float): Lattice parameter.
        pot_file (str): Path to empirical potential to use.
        species (str): Space separated, potential-appropriate chemical symbols who will be applied (in order) to the
                       type column of the input file.
        cell_repetitions (int): How many repetitions of the unit cell to allow (in each of x-, y-, and
                                z-directions.) (Default is 1.)
        solute_ids (int or np.ndarray or list): Integer id(s) for which to change species. (Default is None.)
        new_type (int): New species value for solute ids. (Default is 2.)
        input_file (str): File to write LAMMPS input script to. (Default is "in.clustergb.lammps")
        nprocs (int): How many processors to run on. (Default is 1.)

    Returns:
        (*Namespace*) -- Parsed thermodynamics data from the log file.
    """
    lammps = cgb.lammps.LammpsJob(input_file=input_file)
    lammps.add_structure_bulk(xl_type, lattice_constant, cell_repetitions, nspec=len(species.split()))
    lammps.add_potential(pot_file, species)
    if solute_ids is not None:
        lammps.add_species_change(solute_ids, new_type=new_type)
    lammps.add_dump_thermo()
    lammps.add_run_static()
    lammps.run(nprocs)
    return lammps.thermo_data


class LammpsJob:
    """
    Handles LAMMPS calculations by building a LAMMPS input file through initialization and a series of `add` methods,
    then `run` the LAMMPS executable and parsing the log file.

    Uses LAMMPS 'metal' units, so distances are all in angstroms, time in ps, temperatures in K, and energy in eV.
    """

    def __init__(self, input_file='in.clustergb.lammps', log_file='log.lammps', cluster=None):
        """
        Sets initial parameters for the input file, e.g. units (metal), dimension (3), atoms style (atomic) and log.
        Also sets the intial thermodynamcis keys to track. Finds LAMMPS and MPI executables from the ClusterGB config
        file.

        Args:
            input_file (str): File to write LAMMPS input script to. (Default is "in.clustergb.lammps")
            log_file (str): File for LAMMPS to write its log to. (Default is "log.lammps")
            cluster (tuple): Containing a :math:`(3,)` *np.ndarray* box center and a *float* radius. (Default is None,
                             which triggers only the basic thermodynamic data to be recorded and not cluster info.)
        """

        # Set basic input
        self.header_str = '''# clustergb LAMMPS input file

        units metal
        dimension 3
        atom_style atomic
        
        log {log}
        '''.format(**{'log': log_file})

        self.input_file = input_file
        self.log_file = log_file

        self.input_string = self.header_str
        self.thermo_keys = ['step', 'press', 'temp', 'vol', 'lx', 'ly', 'lz', 'pe', 'ke', 'etotal']
        self.thermo_data = None  # Will be read after we run
        self.has_thermo = False

        self.cluster = cluster
        self.center = None
        self.radius = None
        if cluster is not None:
            self.thermo_keys += ['v_n_all', 'v_n_inner', 'c_pe_inner']
            self.center = cluster[0]
            self.radius = cluster[1]

        # Load executable paths from the config file one level up
        up_level = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(up_level, 'config.yml'), 'r') as f:
            config = yaml.load(f)

        self.lmp_exec = config['executables']['lammps']
        self.mpi_exec = config['executables']['mpi']

    def add_cell_relax(self, pressure=0, isotropy='iso'):
        """
        Add a relaxation command to the LAMMPs input string.

        Args:
            pressure (float): Pressure at which to run in bars (default is 0.)
            isotropy (str): Allow box to relax 'iso'tropically or 'aniso'tropically (orthogonal only.) (Default is iso.)
        """

        if isotropy not in ['iso', 'aniso']:
            raise Exception('Cell relaxation must be "iso" or "aniso".')

        isobar_string = '''# Relax the box
        
        fix isobaric all box/relax {isotropy} {pressure}
        
        '''
        context = {'pressure': pressure, 'isotropy': isotropy}

        self.input_string += isobar_string.format(**context)

    def add_dump_cna(self, xl_type, lattice_constant, dump_file, dump_period=1):
        """
        Add a dump command for common neighbour analysis values for all atoms.

        Args:
            xl_type (str): Crystal structure indicator.
            lattice_constant (float): Lattice parameter.
            dump_file (str): Where to dump the per-atom CNA data.
            dump_period (int): Period (steps) to dump the data. (Default is 1--dump every step.)

        Raises:
                NotImplementedError: If `lattice_type` is "hcp"
                ValueError: If `lattice_type` is not "fcc", "bcc", or "hcp".
        """

        if xl_type == 'fcc':
            cut = 0.5 * ((0.5 * np.sqrt(2)) + 1) * lattice_constant
        elif xl_type == 'bcc':
            cut = 0.5 * (np.sqrt(2) + 1) * lattice_constant
        elif xl_type == 'hcp':
            raise NotImplementedError
            # cut = 0.5 * (1 + np.sqrt((4 + (2*(c/a)**2)) / 2.) ) * a # Needs both c and a lengths
        else:
            raise ValueError(xl_type + ' is not a recognized lattice')

        cna_string = '''# Compute CNA
        compute cna_vals all cna/atom {cna_cut}
        dump cna_dump all custom {period} {cna_dump} c_cna_vals
        dump_modify cna_dump sort id

        '''

        context = {'cna_cut': cut, 'period': dump_period, 'cna_dump': dump_file}

        self.input_string += cna_string.format(**context)

    def add_dump_thermo(self, dump_period=1):
        """
        Add a dump command for basic thermo properties (c.f. `self.thermo_keys`), and if the cluster option was given
        also dumps spatial data.

        Args:
            dump_period (int): Period (steps) to dump the data. (Default is 1--dump every step.)
        """

        thermo_string = '#Dump thermodynamics'

        if self.cluster is not None:
            cluster_thermo = '''
            region inner_sphere sphere {x} {y} {z} {r} units box
            group inner_atoms region inner_sphere
    
            compute pot_en all pe/atom
            compute pe_inner inner_atoms reduce sum c_pot_en
    
            variable n_all equal count(all)
            variable n_inner equal count(inner_atoms)
            '''
            cluster_context = {'x': self.center[0], 'y': self.center[1], 'z': self.center[2], 'r': self.radius}

            thermo_string += cluster_thermo.format(**cluster_context)

        thermo_string += '''
        thermo {period}
        thermo_style custom {keys}
        thermo_modify line one format float %10.8f

        '''
        context = {'period': dump_period, 'keys': ' '.join(self.thermo_keys)}

        self.input_string += thermo_string.format(**context)
        self.has_thermo = True

    def add_dump_xyz(self, dump_file, dump_period=100):
        """
        Add a dump command for the cartesian positions of the atoms.

        Args:
            dump_file (str): Where to write the positions to. (Recommended extension for the file is .xyz)
            dump_period (int): Period (steps) to dump the data. (Default is 100--dump every hundredth step.)
        """

        dump_string = '''# Dump xyz positions
        dump posdump all xyz {period} {file}

        '''

        context = {'file': dump_file, 'period': dump_period}

        self.input_string += dump_string.format(**context)

    def add_potential(self, pot_file, species):
        """
        Add commands for which potential to use and which species to reference within it.

        Args:
            pot_file (str): Path to empirical potential to use.
            species (str): Space separated, potential-appropriate chemical symbols who will be applied (in order) to
                           the type column of the input file.
        """

        style = _potential_style(pot_file)

        potential_string = '''
        atom_modify	sort 0 1.0
        pair_style {style}
        pair_coeff * * {potential} {species}

        '''

        context = {'potential': pot_file, 'style': style, 'species': species}

        self.input_string += potential_string.format(**context)

    def add_print_value(self, value, filename=None):
        """
        Add a command to print a system reserved value. Keywords can be found at the LAMMPS documentation_.

        .. _documentation: http://lammps.sandia.gov/doc/thermo_style.html

        Args:
            value (str): LAMMPS tag for the value to print.
            filename (str): Where to write the value to. (Default is None--print only to screen)

        Raises:
            Exception: If `value` is not a valid LAMMPS keyword.
        """

        value = value.lower()

        possible_keywords = ['step', 'elapsed', 'elaplong', 'dt', 'time',
                             'cpu', 'tpcpu', 'spcpu', 'cpuremain', 'part', 'timeremain',
                             'atoms', 'temp', 'press', 'pe', 'ke', 'etotal', 'enthalpy',
                             'evdwl', 'ecoul', 'epair', 'ebond', 'eangle', 'edihed', 'eimp',
                             'emol', 'elong', 'etail',
                             'vol', 'density', 'lx', 'ly', 'lz', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi',
                             'xy', 'xz', 'yz', 'xlat', 'ylat', 'zlat',
                             'bonds', 'angles', 'dihedrals', 'impropers',
                             'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz',
                             'fmax', 'fnorm', 'nbuild', 'ndanger',
                             'cella', 'cellb', 'cellc', 'cellalpha', 'cellbeta', 'cellgamma']
        if value not in possible_keywords:
            raise Exception('Can only print valid LAMMPS thermo keywords.')

        print_str = '\nvariable printed_' + value + ' equal ' + value + '\n'
        if filename is None:
            print_str += 'print ${printed_' + value + '}\n'
        else:
            print_str += 'print ${printed_' + value + '} file ' + filename + '\n'

        self.input_string += print_str

    def add_restart(self, read_restart):
        """
        A a command for restarting the calculation from a LAMMPS restart file.

        Args:
            read_restart (str): Where to find the LAMMPS restart file.
        """
        restart_string = '''#Restart from a saved run
        read_restart {restart}

        '''
        context = {'restart': read_restart}

        self.input_string += restart_string.format(**context)

    def add_run_minimization(self, max_steps, force_convergence):
        """
        Add a command to minimize the atomic forces using conjugate gradient.

        .. note::

            The minimization command requires a `force eval`, but the LAMMPS docs are ambiguous about what counts as a
            force evaluation, so it is just 1000 times the number of steps for now. There is no evidence of this
            causing premature stopping for calculations with up to 700k atoms.

        Args:
            max_steps (int): Maximum number of conjugate gradient steps to take.
            force_convergence (float): Stopping threshold based on the L2 norm of the global force vector.
        """

        max_evals = 1000 * max_steps

        minimization_string = '''# Minimize
        velocity all set 0. 0. 0.
        min_style cg
        minimize 0.0 {fconv} {steps} {evals}

        '''

        context = {'fconv': force_convergence, 'steps': max_steps, 'evals': max_evals}

        self.input_string += minimization_string.format(**context)

    def add_run_nve_damped(self, steps, timestep, damping):
        """
        Add commands to run the system with an NVE (microcanoncical) integrator and apply damping friction to the atoms.

        Args:
            steps (float): How long to run for.
            timestep (float): Velocity verlet integration time step size.
            damping (float): Coefficient of friction.
        """

        quench_string = '''# Quench
        
        timestep {timestep}
        fix nve_integrator all nve
        fix damp_motion all viscous {damping}

        run {steps}
        
        unfix damp_motion
        unfix nve_integrator
        '''

        context = {'timestep': timestep, 'damping': damping, 'steps': steps}

        self.input_string += quench_string.format(**context)

    def add_run_static(self):
        """
        Add commands to run the system for zero steps. Useful for forcing output without changing the system, e.g.
        with CNA.
        """

        run_string = '''# \"Run\"
        fix nve_integrator all nve
        run 0
        unfix nve_integrator

        '''

        self.input_string += run_string

    def add_run_nvt_langevin(self, steps, timestep, temp, langevin_period,
                             init_temp=None, seed=None, write_restart=None):
        """
        Add commands to run the system with an NVT (canonical) integrator using a Langevin thermostat to equilibrate
        temperature.

        Args:
            steps (float): How long to run for.
            timestep (float): Velocity verlet integration time step size.
            temp (float): Target temperature for the thermostat.
            langevin_period (float): Parameter to control Langevin damping. Long times give weak damping.
            init_temp (float): Temperature for initial velocity distribution. (Default is `temp`.)
            seed (int): Seed to use when generating random initial velocities. (Default is random.)
            write_restart (str): Location to write a restart file to. (Default is to not write a file.)
        """

        if seed is None:
            seed = np.random.randint(low=1, high=65536)  # I'm just capping the seed at 2^16

        nvt_string = '# Run NVT with Langevin damping\ntimestep {timestep}\n'
        if init_temp is not None:
            nvt_string += 'velocity all create {temp0} {seed} mom yes rot yes dist gaussian\n'
        else:
            nvt_string += 'velocity all create {temp} {seed} mom yes rot yes dist gaussian\n'

        nvt_string += '''
        fix nve_integrator all nve
        fix lang_thermostat all langevin {temp} {temp} {langevin} {seed}

        run {steps}
        unfix lang_thermostat
        unfix nve_integrator
        
        '''

        if write_restart is not None:
            nvt_string += 'write_restart ' + write_restart + '\n\n'

        context = {'timestep': timestep, 'temp0': init_temp, 'seed': seed, 'temp': temp, 'langevin': langevin_period,
                   'steps': steps}

        self.input_string += nvt_string.format(**context)

    def add_species_change(self, ids, new_type=2):
        """
        Add a command to change the species of some of the atoms.

        .. warning::

            If `new_type` exceeds the number of species available in the structure file that was read, this will cause
            a failure in the LAMMPS run.

        Args:
            ids (int or np.ndarray or list): Integer id(s) for which to change species.
            new_type (int): New species value for given ids. (Default is 2.)
        """

        solute_string = ''
        ids += 1  # Because LAMMPS starts counting at 1, but we start counting at 0
        try:
            for n in ids:
                solute_string += 'set atom ' + str(int(n)) + ' type ' + str(new_type) + '\n'
        except TypeError:
            solute_string += 'set atom ' + str(int(ids)) + ' type ' + str(new_type) + '\n'

        self.input_string += solute_string

    def add_structure_bulk(self, xl_type, lattice_constant, cell_repetitions, nspec=1):
        """
        Add a command to set the structure to a block of the ideal crystal.

        Args:
            xl_type (str): Crystal structure indicator.
            lattice_constant (float): Lattice parameter.
            cell_repetitions (int): How many repetitions of the unit cell to allow (in each of x-, y-, and
                                    z-directions.)
            nspec (int): How many species to allow. (Default is 1.)
        """

        xl_type = xl_type.lower()
        if xl_type not in ['fcc', 'bcc']:
            raise Exception('Only "fcc" and "bcc" unit cells may be generated.')

        unit_cell_string = '''# Build a unit cell of {xl_type}

        lattice {xl_type} {latt} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1
        region box block 0 {reps} 0 {reps} 0 {reps} units lattice
        create_box {nspec} box

        create_atoms 1 region box

        '''
        context = {'xl_type': xl_type, 'latt': lattice_constant, 'reps': cell_repetitions,
                   'nspec': nspec}

        self.input_string += unit_cell_string.format(**context)

    def add_structure_file(self, structure_file):
        """
        Add a command to read the structure from a file.

        Args:
            structure_file (str): Path to .xyzin formatted structure file to read.
        """

        structure_string = '''# Read the structure
        read_data {struct}

        '''

        context = {'struct': structure_file}

        self.input_string += structure_string.format(**context)

    def add_structure_single(self, vacuum_distance=25.):
        """
        Add a command to build the structure as a single atoms surrounded by vacuum.

        Args:
            vacuum_distance (float): The vacuum distance between periodic images, i.e. the supercell edge length.
                                     (Default is 25.)
        """

        unit_cell_string = '''# Build a box with a single atom

        lattice fcc {dist}
        region box block 0 1 0 1 0 1 units lattice
        create_box 1 box

        create_atoms 1 single 0. 0. 0.

        '''
        context = {'dist': vacuum_distance}

        self.input_string += unit_cell_string.format(**context)

    def add_vacancy(self, ids):
        """
        Delete some of the atoms.

        :param int or numpy.array(n, dtype=int) or list(int) ids: Which id(s) to turn to vacancies.
        :return:
        """

        ids += 1  # Because LAMMPS starts counting at 1, but we start counting at 0
        vac_string = 'group vacancy_group id '
        try:
            for n in ids:
                vac_string += str(n)
        except TypeError:  # Thrown if ids was an int instead of list or array
            vac_string += str(ids)
        vac_string += '\ndelete_atoms group vacancy_group\n'

        self.input_string += vac_string

    def run(self, nprocs):
        """
        Writes the `input_string` to file and executes LAMMPS (using MPI if available and requested.)

        .. todo::

            Make one processor the default.

        Args:
            nprocs (int): How many processors to run on.
        """

        # Write LAMMPS script
        with open(self.input_file, 'w') as f:
            f.write(self.input_string)

        # Build the shell command string
        mpi_string = _mpi_command(nprocs, self.mpi_exec)
        command = mpi_string + self.lmp_exec + ' -in ./' + self.input_file

        # Execute
        try:
            with open(os.devnull, 'w') as fnull:
                sp.check_call([command], shell=True, stdout=fnull, stderr=sp.STDOUT)
        except sp.CalledProcessError:
            raise sp.CalledProcessError("Something went wrong with the LAMMPS execution. "
                                        "Check LAMMPS input and output files.")
        except:
            raise

        # Parse LAMMPS log file
        if self.has_thermo:
            self._parse_thermo()

    def _parse_thermo(self):
        """
        Reads thermo data in the objects LAMMPS log file.

        .. todo::

            Lock the cluster variable names for reading to the names used for writing that data to prevent a failure if
            one of them gets changed.
        """

        thermo_header = 'Step Press Temp Volume Lx Ly Lz PotEng KinEng TotEng'  # Always added to thermo
        header_found = False

        first_data = None
        last_line = None
        with open(self.log_file, 'r') as f:
            for line in f:
                if (first_data is None) and header_found:
                    first_data = line.split()
                if thermo_header in line:
                    header_found = True
                if (last_line is not None) and ('Loop time of' in line) and header_found:
                    data = last_line.split()
                    break
                last_line = line

        # Reformat all the data from strings to numbers
        thermo_data = {'init_vals': {}}
        for i, key in enumerate(self.thermo_keys):
            thermo_data[key] = float(data[i])
            thermo_data['init_vals'][key] = float(first_data[i])

        # If we've got cluster data, turn the atom counts into integers
        if self.cluster is not None:
            thermo_data['v_n_all'] = int(thermo_data['v_n_all'])
            thermo_data['v_n_inner'] = int(thermo_data['v_n_inner'])

        thermo_data['init_vals'] = argparse.Namespace(**thermo_data['init_vals'])

        self.thermo_data = argparse.Namespace(**thermo_data)


def _mpi_command(nprocs, mpi_exec):
    """
    Generate a string for shell execution of MPI. If the number of processors is 1 or the executable path is empty,
    the returned value is simply an empty string.

    Args:
        nprocs (int): Number of processors to use.
        mpi_exec (str): Path to MPI executable.

    Returns:
        (*str*) -- Shell-compatible code for executing MPI.
    """
    if nprocs == 1 or mpi_exec == '':
        return ''
    else:
        return mpi_exec + ' -np ' + str(nprocs) + ' '


def _potential_style(potential):
    """
    Determine which style of EAM your potential uses, e.g. 'alloy', 'fs'.

    Args:
        potential (str): Path to an EAM potential file.

    Returns:
        (*str*) -- The LAMMPS "style" of EAM that the potential has.

    Raises:
        NotImplementedError: If the potential file doesn't end in ".fs" or ".alloy".
    """

    potential_terminus = potential.split('.')[-1]
    if potential_terminus == 'fs':
        style = 'eam/fs'
    elif potential_terminus == 'alloy':
        style = 'eam/alloy'
    else:
        raise NotImplementedError("The potentials style " + potential_terminus + " is not implemented.")

    return style
