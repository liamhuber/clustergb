#!/usr/bin/env python -i
#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Launches an interactive python session with useful libraries loaded, e.g. the clustergb package. Great for exploring
your results, for example:

.. code-block:: bash

    cd ~/my_cgb_project
    cgb_interactive.py

    >>> proj = cgb.osio.load_project()
    >>> for cn in proj.child_names:
    ...     job = cgb.osio.load_job(cn)
    ...     print(cn, job.gb.gb_energy)
    ...
    (gb1, 503.248)
    (gb2, 397.274)
    (gb3, 440.119)
"""

from __future__ import absolute_import
import numpy as np
#from . import clustergb as cgb
import clustergb as cgb
import readline
import rlcompleter
import os
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"


readline.parse_and_bind("tab: complete")
