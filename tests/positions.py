import numpy as np


def create_positions(structure):
    return globals()['get_positions_' + structure]()


def get_positions_zero():
    return np.zeros((0, 3))


def get_positions_one():
    positions = [
        [0, 0, 1],
    ]
    return np.array(positions)


def get_positions_line():
    positions = [
        [0, 0,  1],
        [0, 0, -1],
    ]
    return np.array(positions)


def get_positions_triangle():
    p0 = 0.5 * np.sqrt(3.0)
    positions = [
        [ 1.0, 0.0, 0.0],
        [-0.5,  p0, 0.0],
        [-0.5, -p0, 0.0],
    ]
    return np.array(positions)


def get_positions_square():
    positions = [
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
    ]
    return np.array(positions)


def get_positions_pentagon():
    positions = [
        [np.cos(0.0 * np.pi), np.sin(0.0 * np.pi), 0.0],
        [np.cos(0.4 * np.pi), np.sin(0.4 * np.pi), 0.0],
        [np.cos(0.8 * np.pi), np.sin(0.8 * np.pi), 0.0],
        [np.cos(1.2 * np.pi), np.sin(1.2 * np.pi), 0.0],
        [np.cos(1.6 * np.pi), np.sin(1.6 * np.pi), 0.0],
    ]
    return np.array(positions)


def get_positions_hexagon():
    p0 = 0.5 * np.sqrt(3.0)
    positions = [
        [ 1.0, 0.0, 0.0],
        [ 0.5,  p0, 0.0],
        [-0.5,  p0, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.5, -p0, 0.0],
        [ 0.5, -p0, 0.0],
    ]
    return np.array(positions)


def get_positions_tetrahedron():
    positions = [
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ]
    r = np.sqrt(3.0)
    return np.array(positions) / r


def get_positions_cube():
    positions = [
        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [-1, -1, -1],
    ]
    r = np.sqrt(3.0)
    return np.array(positions) / r


def get_positions_octahedron():
    positions = [
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
        ]
    return np.array(positions)


def get_positions_fcc():
    p0 = 0.5 * np.sqrt(3.0)
    p2 = np.sqrt(2.0 / 3.0)
    positions = [
        [ 1.0, 0.0, 0.0],
        [ 0.5,  p0, 0.0],
        [-0.5,  p0, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.5, -p0, 0.0],
        [ 0.5, -p0, 0.0],
        #
        [ 0.5, -np.sqrt(3.0) / 6.0,  p2],
        [ 0.0,  np.sqrt(3.0) / 3.0,  p2],
        [-0.5, -np.sqrt(3.0) / 6.0,  p2],
        [ 0.5,  np.sqrt(3.0) / 6.0, -p2],
        [ 0.0, -np.sqrt(3.0) / 3.0, -p2],
        [-0.5,  np.sqrt(3.0) / 6.0, -p2],
        ]
    return np.array(positions)


def get_positions_hcp():
    p0 = 0.5 * np.sqrt(3.0)
    p2 = np.sqrt(2.0 / 3.0)
    positions = [
        [ 1.0, 0.0, 0.0],
        [ 0.5,  p0, 0.0],
        [-0.5,  p0, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.5, -p0, 0.0],
        [ 0.5, -p0, 0.0],
        #
        [ 0.5, -np.sqrt(3.0) / 6.0,  p2],
        [ 0.0,  np.sqrt(3.0) / 3.0,  p2],
        [-0.5, -np.sqrt(3.0) / 6.0,  p2],
        [ 0.5, -np.sqrt(3.0) / 6.0, -p2],
        [ 0.0,  np.sqrt(3.0) / 3.0, -p2],
        [-0.5, -np.sqrt(3.0) / 6.0, -p2],
    ]
    return np.array(positions)


def get_positions_dodecahedron():
    """Return the edges of the dodecahedron.

    We first make the edges of an icosahedron with the edge length of
    :math:`\sqrt{3}`.

    https://en.wikipedia.org/wiki/Regular_dodecahedron#Cartesian_coordinates

    We then scale it by the distance from the center.
    """
    phi0 = (np.sqrt(5.0) + 1.0) * 0.5
    phi1 = (np.sqrt(5.0) - 1.0) * 0.5  # 1.0 / phi0
    positions = [
        [ 1.0,  1.0,  1.0],
        [ 1.0,  1.0, -1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [-1.0, -1.0, -1.0],
        #
        [ 0.0, +phi0, +phi1],
        [ 0.0, +phi0, -phi1],
        [ 0.0, -phi0, +phi1],
        [ 0.0, -phi0, -phi1],
        #
        [+phi1,  0.0, +phi0],
        [-phi1,  0.0, +phi0],
        [+phi1,  0.0, -phi0],
        [-phi1,  0.0, -phi0],
        #
        [+phi0, +phi1,  0.0],
        [+phi0, -phi1,  0.0],
        [-phi0, +phi1,  0.0],
        [-phi0, -phi1,  0.0],
    ]
    r = np.sqrt(3.0)
    return np.array(positions) / r


def get_positions_icosahedron():
    """Return the edges of the icosahedron.

    We first make the edges of an icosahedron with the edge length of 2.

    https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates

    We then scale it by the distance from the center.
    """
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    positions = [
        [ 0.0,  1.0,  phi],
        [ 0.0,  1.0, -phi],
        [ 0.0, -1.0,  phi],
        [ 0.0, -1.0, -phi],
        #
        [ 1.0,  phi,  0.0],
        [ 1.0, -phi,  0.0],
        [-1.0,  phi,  0.0],
        [-1.0, -phi,  0.0],
        #
        [ phi,  0.0,  1.0],
        [-phi,  0.0,  1.0],
        [ phi,  0.0, -1.0],
        [-phi,  0.0, -1.0],
    ]
    r = np.sqrt(22.0 + 2.0 * np.sqrt(5.0)) * 0.25
    return np.array(positions) / r


def get_positions_cuboctahedron():
    """Return the edges of the cuboctahedron.

    We first make the edges of a cuboctahedron with the edge length of
    :math:`\sqrt(2)`.

    https://en.wikipedia.org/wiki/Cuboctahedron#Cartesian_coordinates

    We then scale it by the distance from the center.

    This should be the same as the fcc coordination.
    """
    positions = [
        [ 1,  1,  0],
        [ 1, -1,  0],
        [-1,  1,  0],
        [-1, -1,  0],
        #
        [ 1,  0,  1],
        [ 1,  0, -1],
        [-1,  0,  1],
        [-1,  0, -1],
        #
        [ 0,  1,  1],
        [ 0,  1, -1],
        [ 0, -1,  1],
        [ 0, -1, -1],
    ]
    r = np.sqrt(2.0)
    return np.array(positions) / r
