#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - 2026 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of a library developed by Louis Vanduyfhuys at
# the Center for Molecular Modeling. Usage of this package should be 
# authorized by prof. Van Vanduyfhuys.

# This file has been retrieved from the MolMod package, which is a 
# collection of molecular modelling tools for python.

"""Some useful physicochemical constants in atomic units

   These are the physical constants defined in this module (in atomic units):

"""

boltzmann = 3.1668154051341965e-06
avogadro = 6.0221415e23
lightspeed = 137.03599975303575
planck = 6.2831853071795864769


# automatically spice up the docstrings

lines = [
    "    ================  ==================",
    "    Name              Value             ",
    "    ================  ==================",
]

for key, value in sorted(globals().items()):
    if not isinstance(value, float):
        continue
    lines.append("    %16s  %.10e" % (key, value))
lines.append("    ================  ==================")

__doc__ += "\n".join(lines)
