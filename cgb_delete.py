#!/usr/bin/env python
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Safely deletes Jobs from their parent Projects and the file system.

.. note::

        Tab completing the Job you want to delete will likely put a "/" at the end of the Job name. The directory will
        be deleted, but the Project won't recognize the child name and you will get a warning. Simply re-run without
        the "/"--you will get a warning that the directory is not present (since it's already deleted) but this will
        clean the child name from the Project.
"""

from __future__ import absolute_import
import argparse
#from . import clustergb as cgb
import clustergb as cgb
from shutil import rmtree

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


def main(args):
    try:
        parent = cgb.osio.load_project()
    except IOError:
        raise Exception("You're not in a ClusterGB Project hierarchy, just use rm...")

    for target in args.to_delete:
        try:
            parent.child_names.remove(target)
            parent.save()
        except ValueError:
            print("WARNING: Child " + target + " does not belong to " + parent.__class__.__name__ + " " + parent.name +
                  ".")

        try:
            rmtree(target)
        except OSError:
            print("WARNING: No directory " + target + " was found.")
    return


def _ret_parser():
    parser = argparse.ArgumentParser(description="Delete Jobs by name--including their references in their parent "
                                                 "Projects.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("to_delete", type=str, nargs="+",
                        help="Names of the Jobs/Projects in the current working directory to delete.")

    return parser


if __name__ == "__main__":

    returned_parer = _ret_parser()

    arguments = returned_parer.parse_args()

    main(arguments)
