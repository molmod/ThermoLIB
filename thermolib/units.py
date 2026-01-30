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

# This file has been retrieved from the MolMod package, which is/was a 
# collection of molecular modelling tools for python.

"""Conversion from and to atomic units

   Internally ThermoLIB works always in atomic units. This unit system
   is consistent, like the SI unit system one does not need conversion factors
   in the middle of a computation once all values are converted to atomic units.
   This facilitates the programming and reduces accidental bugs due to
   forgetting these conversion factor in the body of the code.

   References for the conversion values:

   * B. J. Mohr and B. N. Taylor,
     CODATA recommended values of the fundamental physical
     constants: 1998, Rev. Mod. Phys. 72(2), 351 (2000)
   * The NIST Reference on Constants, Units, and Uncertainty
     (http://physics.nist.gov/cuu/Constants/index.html)
   * 1 calorie = 4.184 Joules

   Naming conventions in this module: unit is the value of one external unit
   in internal - i.e. atomic - units. e.g. If you want to have a distance of
   five angstrom in internal units: ``5*angstrom``. If you want to convert a
   length of 5 internal units to angstrom: ``5/angstrom``. It is recommended to
   perform this kind of conversions, only when data is read from the input and
   data is written to the output.

   An often recurring question is how to convert a frequency in internal units
   to a spectroscopic wavenumber in inverse centimeters. This is how it can be
   done::

     >>> from thermolib.units import centimeter
     >>> from thermolib.constants import lightspeed
     >>> invcm = lightspeed/centimeter
     >>> freq = 0.00320232
     >>> print freq/invcm

   These are the conversion constants defined in this module:

"""


from .constants import avogadro

import ast
import operator as op

# define which operations among units are allowed
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def parse_unit(expression):
    """Evaluate a python expression string containing constants

       Argument:
        | ``expression``  --  A string containing a numerical expressions
                              including unit conversions.
    """
    try:
        node = ast.parse(expression, mode='eval').body
        g = globals()
        g.update(shorthands)
        return _eval_ast(node, g)
    except:
        raise ValueError(f"Invalid expression '{expression}'")
    

def _eval_ast(node, g):
    '''Recursive evaluation of the string'''
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        # evaluate a binary operation (e.g., a + b)
        left = _eval_ast(node.left, g)
        right = _eval_ast(node.right, g)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left, right)
        else:
            raise ValueError(f"Invalid operation '{op_type}'")
    elif isinstance(node, ast.UnaryOp):
        # evaluate a unary operation (e.g., -a)
        operand = _eval_ast(node.operand, g)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Invalid operation '{op_type}'")
    elif isinstance(node, ast.Name):
        # replace variable name with its value
        if node.id in g:
            return g[node.id]
        else:
            raise ValueError(f"Unknown unit '{node.id}'")
    else:
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


# *** Generic ***
au = 1.0


# *** Charge ***

coulomb = 1.0/1.602176462e-19

# Mol

mol = avogadro

# *** Mass ***

kilogram = 1.0/9.10938188e-31

gram = 1.0e-3*kilogram
milligram = 1.0e-6*kilogram
unified = 1.0e-3*kilogram/mol
amu = unified

# *** Length ***

meter = 1.0/0.5291772083e-10

decimeter = 1.0e-1*meter
centimeter = 1.0e-2*meter
millimeter = 1.0e-3*meter
micrometer = 1.0e-6*meter
nanometer = 1.0e-9*meter
angstrom = 1.0e-10*meter
picometer = 1.0e-12*meter

# *** Volume ***

liter = decimeter**3

# *** Energy ***

joule = 1/4.35974381e-18

calorie = 4.184*joule
kjmol = 1.0e3*joule/mol
kcalmol = 1.0e3*calorie/mol
electronvolt = (1.0/coulomb)*joule
rydberg = 0.5

# *** Force ***

newton = joule/meter

# *** Angles ***

deg = 0.017453292519943295
rad = 1.0

# *** Time ***

second = 1/2.418884326500e-17

nanosecond = 1e-9*second
femtosecond = 1e-15*second
picosecond = 1e-12*second

# *** Frequency ***

hertz = 1/second

# *** Pressure ***

pascal = newton/meter**2
bar = 100000*pascal
atm = 1.01325*bar

# *** Temperature ***

kelvin = 1.0

# *** Dipole ***

debye = 0.39343031369146675 # = 1e-21*coulomb*meter**2/second/lightspeed

# *** Current ***

ampere = coulomb/second


# Shorthands for the parse functions

shorthands = {
    "C": coulomb,
    "kg": kilogram,
    "g": gram,
    "mg": milligram,
    "u": unified,
    "m": meter,
    "cm": centimeter,
    "mm": millimeter,
    "um": micrometer,
    "nm": nanometer,
    "A": angstrom,
    "pm": picometer,
    "l": liter,
    "J": joule,
    "cal": calorie,
    "eV": electronvolt,
    "N": newton,
    "s": second,
    "Hz": hertz,
    "ns": nanosecond,
    "fs": femtosecond,
    "ps": picosecond,
    "Pa": pascal,
    "K": kelvin,
    # atomic units
    "e": au,
}


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


lines = [
    "     ================  ==================",
    "         Short name        Value             ",
    "         ================  ==================",
]

for key, value in sorted(shorthands.items()):
    if not isinstance(value, float):
        continue
    lines.append("         %16s  %.10e" % (key, value))
lines.append("         ================  ==================")

parse_unit.__doc__ += "\n".join(lines)

del lines
