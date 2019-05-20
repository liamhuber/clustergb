import numpy as np
import pandas as pd
from clustergb.bond_orientational_order import (
    bond_orientational_order_parameter)
from positions import create_positions
import pytest


df = pd.read_csv(
    'bond_orientational_order.dat',
    delim_whitespace=True,
    index_col='structure',
)

structures = [
    # 'zero',
    'one',
    'line',
    'triangle',
    'square',
    # 'pentagon',
    'hexagon',
    'tetrahedron',
    'cube',
    'octahedron',
    'dodecahedron',
    'icosahedron',
    'fcc',
    'hcp',
    'cuboctahedron',
]


@pytest.mark.parametrize('structure', structures)
def test_values(structure):
    positions = create_positions(structure)
    lmax = 16
    qs = run(positions, lmax)

    keys = ['Q_{}'.format(i + 1) for i in range(lmax)]
    qs_expected = df.loc[structure][keys]

    assert np.allclose(qs, qs_expected)


def run(positions, lmax):
    # Add atom on the center. Bond order is analyzed on this center atom.
    pos = np.vstack(([0, 0, 0], positions))
    # disps = ps[None, :, :] - ps[:, None, :]
    rcut = 2.0
    return bond_orientational_order_parameter(
        pos, [0], xl_structure=None, latt=None, lmax=lmax, rcut=rcut)
