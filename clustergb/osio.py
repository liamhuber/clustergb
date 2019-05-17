#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Functions for os and io.
"""

import os
import pickle
import logging
from . import project
from . import job
import subprocess as sp

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def count_lines(fin):
    """
    Count the number of lines in a file. Python `grep ${fin} | wc -l` replacement.

    Args:
        fin (str): Path to file.

    Returns:
        (*int*) -- Number of lines in file.
    """
    with open(fin, 'r') as f:
        n = 0
        for _ in f:
            n += 1
    return n


def file_extension(fpath):
    """
    Get the extension associated with a  file, i.e. the bit after the final '.'.

    Args:
        fpath (str): Path to the file in question.

    Returns:
        (*str*) Just the extension.
    """

    if not os.path.exists(fpath):
        raise ValueError
    return fpath.split('.')[-1]


def load(filename, path='.'):
    """Load a file as a pickled object and return it."""
    with open(os.path.join(path, filename), 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_job(path='.'): return load(job.Job.job_save_name, path)


def load_project(path='.'): return load(project.Project.proj_save_name, path)


def make_dir(name, allow_exists=False):
    """
    Make a new directory in the present location.

    Args:
        name (str): New directory name.
        allow_exists (bool): Whether it's OK if there is already an object with *name* at this location. (Default is
                             False.)

    Raises:
        OSError: If there is already something named *name* here and *allow_exist* is False
    """
    if os.path.isdir(os.path.join('.', name)):
        if allow_exists:
            pass
        else:
            raise OSError('The directory ' + name + ' already exists.')
    else:
        os.makedirs(name)


def run_in_hierarchy(func, kwargs):
    """
    Runs a function for all `Jobs` downstream of this point in the hierarchy.

    Args:
        func (callable): The function to call, which takes as its arguments a `job.Job` and any keyword arguments.
        kwargs (dict): The keyword arguments for `func`.

    Raises:
        Exception: When called from somewhere that doesn't have a *Job* or *Project* object to load.
    """
    try:
        proj_obj = load_project()
        proj_obj.run_in_children(func, kwargs)
    except IOError:  # Look instead for a job
        try:
            job_obj = load_job()
            func(job_obj, **kwargs)
        except IOError:
            raise Exception("You are not inside a ClusterGB hierarchy.")


def run_in_shell(command):
    """Runs a command in the shell."""
    with open(os.devnull, 'w') as fnull:
        sp.check_call([command], shell=True, stdout=fnull, stderr=sp.STDOUT)


def tee(log_loc, message, severity='info'):
    """
    Tees a message string to both stdout and the parent logger.

    Args:
        log_loc (str): Path to the log file to write to.
        message (str): The string to log.
        severity (str): The log severity: 'info', 'warning', or 'error'. (Default is 'info'.)
    """

    if severity not in ['info', 'warning', 'error']:
        raise ValueError('Logging severity must be one of info, warning, or error.')

    # Initialize the logger
    logging.basicConfig(filename=log_loc, level=logging.INFO)
    log = logging.getLogger(__name__)

    # Log and print the message
    exec('log.' + str(severity) + '(message)')
    if severity != "info":
        message = severity.upper() + ": " + message
    print(message)
