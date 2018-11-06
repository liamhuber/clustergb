# Cluster GB

## Introduction

ClusterGB is a set of Python scripts designed to facilitate easy calculations of planar grain boundaries (GBs) using the Large-scale Atomic/Molecular Massively Parallel Simulator ([LAMMPS](http://lammps.sandia.gov)). In order to accommodate even low-symmetry boundaries, ClusterGB uses vacuum clusters to eliminate the need for periodic boundary conditions.

The user first begins a `Project` to specify a material (i.e. an emperical potential), then inside this `Project` creates a series of `Jobs`--one for each GB to be studied.

Initializing a new `Job` generates a GB structure file with the requested GB character and
(stochastically) minimized microscopic degrees of freedom. From here various `cgb_run_*` scripts can be called from either the `Job` or `Project` level of the hierarchy to calculate properties of a particular GB or all the GBs in the project, respectively. An important example is calculating

ClusterGB can accommodate any macroscopic GB character, but currently only works for single-species host structures with FCC or BCC crystal structure, and relies on LAMMPS_ with
[Embedded Atom Method](http://lammps.sandia.gov/doc/pair_eam.html) potentials with "alloy" or "Finnis-Sinclair"
formatting.

For more details, please read the documentation accompanying the source code.

## Installation

ClusterGB has only been tested on Unix systems. Sorry, Windows.

### 0) Environment

ClusterGB was written for Python2.7 and uses a variety of Python packages. If you are already familiar with Python, simply ensure that your environment has all of the packages listed below. If you are unfamiliar with Python, you can download it for free from a variety of places, but I recommend an [Anaconda distribution](https://conda.io/docs/user-guide/install/index.html). With Anaconda, any missing packages can be easily installed with `conda install ${PACKAGE_NAME}`

All required Packages (most scientific installations of Python will already have these):

* `argparse`
* `logging`
* `matplotlib` (just for `cgb_interactive.py`)
* `numpy`
* `os`
* `readline` (just for `cgb_interactive.py`)
* `rlcompleter` (just for `cgb_interactive.py`)
* `scipy`
* `seaborn` (just for `cgb_interactive.py`)
* `shutil`
* `sklearn` (just for the example project)
* `subprocess`
* `time`
* `yaml`

The heavy-lifting of calculating energies and forces for ClusterGB is done using LAMMPS. LAMMPS is invoked as a separate subprocess, so you just need a regular LAMMPS executable, which can be built from the source code or, for some systems, downloaded pre-built [here](http://lammps.sandia.gov/download.html). Most versions of LAMMPS should be fine, but ClusterGB was tested using the 1 Jul 2016 release.

If MPI runs on your machine, it can be used to accelerate the LAMMPS calculations by running them in parallel.

Finally, Voronoi analysis (optional) is performed by calling Voro++, which is also free to [download](http://math.lbl.gov/voro++/download/).

### 1) Download


If you have these docs, you probably already have the source code for ClusterGB. If not, you can download it from [Github](https://github.com/liamhuber/clustergb). Python is an interpreted language, so no special compilation is required, just add the `clustergb` folder (containing `cgb_init_project.py` etc.) to your path for easy use.


### 2) Configuration

In `clustergb/config.yml`, set the `lammps`, (optionally) `voro`, and (optionally) `mpi` fields to point to
valid executables on your machine for LAMMPS, Voro++, and MPI, respectively.

That's it, you're ready to go with ClusterGB.

## Licence

ClusterGB is released under the MIT License.

## Citing

Different functionality of clustergb requires different citations. Please consider which pieces of the code you're using and add citations to your journal article/website accordingly.

__Main code:__

* https://github.com/liamhuber/clustergb

* Huber, Hadian, Grabowski, and Neugebauer, IN PRESS (2018)

* Lee and Choi, Modelling Simul. Mater. Sci. Eng. 12 (2004) 621

__Voronoi analysis__:

(Assuming you've got a copy of `voro++` that you've linked to and are using cgb's default Voro procedures)

* http://math.lbl.gov/voro++/

* Rycroft, Grest, Landry, and Bazant, Phys. Rev. E 74 (2006) 021306

__Coordination analysis__:

* Huang, Grabowski, McEniry, Trinkle, and Neugebauer, Phys. Status Solidi B 252 (2015) 1907

__Bond orientational order parameters__:

* Steinhardt, Nelson, and Ronchetti, Phys Rev B 28 (1983)
