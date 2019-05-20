#####################################################
# This file is a component of ClusterGB             #
# Copyright (c) 2018 Liam Huber                     #
# Released under the MIT License (see distribution) #
#####################################################
"""
Some useful math functions, mostly rotation-related stuff.

.. todo::

    Replace matrix functions with np.linalg methods.
"""

import numpy as np
from .osio import tee

__author__ = "Liam Huber"
__copyright__ = "Copyright 2018, Liam Huber"
__license__ = "MIT"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "Production"

BASE_PREC = 1E-8
BASE_STEPS = 1000


def alignment_matrix(subject, target, precision=BASE_PREC, verbosity=0, max_steps=BASE_STEPS):
    """
    Numerically find the rotation matrix necessary to rotate the `subject` vector to the `target` direction.

    Args:
        subject (np.ndarray): Length-3 vector to rotate.
        target (np.ndarray): Length-3 vector to rotate to.
        precision (float): Mostly for whether two unit vectors are close enough (L2-norm) to be the same.
        verbosity (bool): Whether to print info from Newton's method search.
        max_steps (int): Maximum allowable steps in Newton's method when searching for the rotation matrix.

    Returns:
        3-element tuple containing

        - (*np.ndarray*): 3x3 rotation matrix to bring `subject` into alignment with `target`.
        - (*np.ndarray*): Length-3 axis of rotation.
        - (*float*): Rotation angle (radians).
    """

    subject = l2normalize(subject)
    target = l2normalize(target)

    if vectors_are_parallel(subject, target):
        return np.identity(3), None, 0
    elif vectors_are_antiparallel(subject, target):
        return -np.identity(3), None, np.pi

    perp_vec = l2normalize(np.cross(subject, target))  # Mutually perpendicular to rotate about

    def err_func(theta):
        rot = rotation(theta, perp_vec)
        subject_guess = matvec_mult(rot, subject)
        err = np.linalg.norm(subject_guess - target)
        return err

    solved_theta, _, _ = newton1d(err_func, precision=precision, verbosity=verbosity, max_steps=max_steps)

    rot_mat = rotation(solved_theta, perp_vec)

    return rot_mat, perp_vec, solved_theta


def fermi(r, r0, sigma, safe_data=False):
    """
    Get the value of the Fermi-Dirac_ distribution at `r`,

    .. math::

        \\frac{1}{1 + \exp \\left( \\frac{r - r_0}{\\sigma} \\right)}.

    .. note::

        If vectors are used for `r`/`r_0`/`sigma`, they must have commensurate shapes.


    Args:
        r (np.ndarray or float): Where on the distribution to evaluate.
        r0 (np.ndarray or float): Distribution midpoint (i.e. Fermi level/half-occupancy point.)
        sigma (np.ndarray or float): Smearing width.
        safe_data (bool): Whether data can be safely used without over-/under-flow clamps. (Default is False, use
                          clamping.)

    Returns:
        (*float*) -- Fermi-Dirac distribution value.

    .. _Fermi-Dirac: https://en.wikipedia.org/wiki/Fermi-Dirac_statistics
    """

    arg = (r - r0) / sigma

    vector_data = isinstance(arg, list) or isinstance(arg, np.ndarray)
    if vector_data and not safe_data:
        # Protect against over and underflow
        max_val = arg > 500
        min_val = np.abs(arg) < 10**(-12)
        negative_max = arg < -1E8

        arg[max_val] = 500
        arg[min_val] = 0.
        arg[negative_max] = -1E8

    return 1. / (1. + np.exp((r - r0) / sigma))


def hypersphere_unit(n, scale=1):
    """
    Get a random vector on the surface of an n-dimensional hypersphere. Credit to Wolfram_ and StackOverflow_.

    .. note:: If `scale` isn't 1, the function name is misleading since it's no longer a unit vector.

    Args:
        n (int): Number of dimensions.
        scale (float): How long the vector should be. (Default is 1, i.e. unit vector.)

    Returns:
        (*np.ndarray*) -- :math:`(n,)` vector randomly sampled from the hypersphere surface.

    .. _Wolfram: http://mathworld.wolfram.com/HyperspherePointPicking.html
    .. _StackOverflow: https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
    """

    # TODO: NOT USED. Destroy?
    rand = np.random.randn(n)
    return scale * rand / np.linalg.norm(rand)


def is_rotation_matrix_3d(mat, precision=BASE_PREC):
    """
    Checks if the provided matrix is a rotation matrix or not (3x3, inverse = transpose, and determinant of +1).

    Args:
        mat (np.ndarray): 3x3 vector to check.
        precision (float): Numeric precision for the inverse and determinant checks. (Default is very small.)

    Returns:
        (*bool*) -- True when the vector meets all three criteria.
    """
    if not mat.shape == (3, 3):
        raise Exception("Expected a 3d rotation matrix, but got a matrix that wasn't 3x3: " + str(mat))
    mtm_sum = np.sum(np.dot(mat.T, mat))
    if np.abs(mtm_sum - 3) > precision:
        raise Exception("A rotation matrix has its transpose = its inverse, this didn't: " + str(mat))
    if np.abs(np.linalg.det(mat) - 1) > precision:
        raise Exception("A rotation matrix should have determinant +1, this didn't: " + str(mat))
    return True


def l2normalize(vec):
    """
    Rescales a vector to unit length.

    Args:
        vec (np.ndarray): Vector to normalize.

    Returns:
        (*np.ndarray*) -- Renormalized `vec` with L2 norm of 1.
    """
    return vec / np.linalg.norm(vec)


def matmat_mult(m1, m2):
    """
    Multiplies two row-major arrays as though they were matrices.

    Args:
        m1 (np.ndarray): 3x3 left matrix.
        m2 (np.ndarray): 3x3 right matrix.

    Returns:
        (*np.ndarray*) -- 3x3 matrix product.
    """

    # TODO: Look at np.inner and see if you can just replace this with a library function
    mat1 = np.matrix(m1)
    mat2 = np.matrix(m2)
    return np.array(np.dot(mat1, mat2))


def matvec_mult(m, v):
    """
    Multiplies a row-major array by a vector as though they were matrices.

    Args:
        m (np.ndarray): :math:`(3,3)` Rotation matrix.
        v (np.ndarray): :math:`(n,3)` vector of real-space vectors to rotate.

    Returns:
        (*np.ndarray*) -- :math:`(n,3)` vector of rotated real-space vectors.
    """

    # TODO: Look at np.inner and see if you can just replace this with a library function
    mat = np.matrix(m)

    # Vectors with a single dimension will need to be reshaped
    # The vector will always need to be transposed to get the inner dimensionality of M.v working
    linear = len(v.shape) == 1
    if linear:
        newv = v.reshape(1, -1).transpose()
    else:
        newv = v.transpose()

    vec = np.matrix(newv)
    multiplied = np.array(np.dot(mat, vec).transpose())

    if linear:
        return multiplied[0]  # [0] gets us back from a (3,1) vector to simply (3,)
    else:
        return multiplied


def newton1d(func, guess=0., step_size=0.1, precision=BASE_PREC, verbosity=0, max_steps=BASE_STEPS):
    """
    Zeroing a function using newton's method for a single dimension.

    Args:
        func (function): Takes a single float and has a zero somewhere you want to find.
        guess (float): Initial guess for where the zero is.
        step_size (float): Initial step size for calculating linear approximation to gradient.
        precision (float): Numeric precision for, e.g. whether we're at zero. Also lower bounds the step_size.
        verbosity (bool): Whether to print information on how the search is going.
        max_steps (int): Maximum allowable iterations of Newton's method.

    Returns:
        3-element tuple

        - (*float*): Solution (i.e. location of zero).
        - (*int*): Number of iterations used.
        - (*float*): Final step size for evaluating gradient
    """

    iters = 0
    f = func(guess)

    while abs(f) > precision:
        if verbosity:
            if iters % verbosity == 0:
                tee("Step", iters, ": x, f(x), dx = ", guess, f, step_size)

        stepped_f = func(guess + step_size)
        df = (stepped_f - f) / step_size  # Linear approx. to gradient
        dguess = f / df
        guess -= dguess  # Thanks, Newton
        f = func(guess)
        # Update the step size so we don't overshoot near a local minima/maxima
        step_size = max(precision, min(step_size, 0.1 * abs(dguess)))

        iters += 1
        if iters > max_steps:
            raise Exception('Newton\'s method found no solution. Final value of '+ str(guess) + ' at step ' +
                            str(max_steps) + ' with a function value of ' + str(f) + '.')

    return guess, iters, step_size


def pole_projection_z(points):
    """
    Project a set of unit vectors onto a unit circle in the (001) plane.

    Args:
        points (np.ndarray): :math:`(n, 3)` vector of 3D points to project. All should have z>=0.

    Returns:
        (*np.ndarray*) -- :math:`(n, 3)` vector of those points (rescaled to unit) and projected onto a circle.
    """

    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
    if np.any(points[:, -1] < 0):
        raise Exception("Can only project points on the northern hemisphere.")

    projected_points = points[:, :2]
    projected_points /= (1. + points[:, -1][:, np.newaxis])

    return projected_points


def rademacher(n, size=None):
    """
    Generates a Rademacher distribution, which is 1's and -1's with equal probability. If scale is given, the magnitude
    of the resulting vector is set to match.

    Args:
        n (int): Dimension (length) of vector.
        size (float): L2-norm length of the resulting vector. (Default is 1.)

    Returns:
        (*np.ndarray*) -- Length-:math:`n` random vector of 1's and -1's with equal probability (rescaled so total
        vector has an L2 norm of `scale`).
    """

    rad = 2 * (np.random.randint(0, 2, n) - 0.5)
    if size is None:
        return rad
    return size * rad / np.linalg.norm(rad)


def rotation(theta, axis):
    """
    Builds a rotation matrix for rotating about the given axis by a set amount. Thanks, Wikipedia and Hamilton.

    Args:
        theta (float): Radians by which to rotate.
        axis (np.ndarray): Length-3 axis about which to rotate.

    Returns:
        (*np.ndarray*) -- :math:`(3,3)` rotation matrix which will appropriately rotate a column vector.
    """

    # I'm getting underflow problems, so just catch it manually. The computational cost won't hurt in this application.
    if abs(theta) < 1E-52:
        theta = 0

    c = np.cos(theta)
    s = np.sin(theta)

    ax = axis[0]
    ay = axis[1]
    az = axis[2]
    rotate = np.matrix([[c + ax * ax * (1 - c),      ax * ay * (1 - c) - az * s, ax * az * (1 - c) + ay * s],
                        [ay * ax * (1 - c) + az * s, c + ay * ay * (1 - c),      ay * az * (1 - c) - ax * s],
                        [az * ax * (1 - c) - ay * s, az * ay * (1 - c) + ax * s, c + az * az * (1 - c)]])
    return np.array(rotate)


def shortest_vector(array):
    """
    Find the shortest vector in a 2D array and get its length.

    Args:
        array (np.ndarray): With shape :math:`(n, m)`.

    Returns:
        (*np.ndarray*) -- Length of the shortest :math:`(1, m)` vector from among the :math:`n` choices.
    """
    if len(array.shape) != 2:
        raise ValueError('Must search over a 2D array/matrix.')
    shortest = np.inf
    for v in array:
        length = np.linalg.norm(v)
        if length < shortest:
            shortest = length
    return shortest


def sigmoid(x, left_asymptote=0, right_asymptote=1, center=0):
    """
    Get the value of a sigmoidal_ function at `x`,

    .. math::

        \\frac{(R - L)}{1 + \exp (-(x - c))} + \\frac{L + R - 1}{2}.

    Args:
        x (float or np.ndarray): Where on the sigmoid to evaluate.
        left_asymptote (float): Limit as `x` is very negative.
        right_asymptote (float): Limit as `x` is very positive.
        center (float): `x` value for the midpoint of transition between the two asymptotes.

    Returns:
        (*float*) Sigmoid evaluated at `x`.

    .. todo::

        Proof the function against over-/under-flow, like the fermi function.

    .. _sigmoidal: https://en.wikipedia.org/wiki/Sigmoid_function
    """
    scale = right_asymptote - left_asymptote
    shift = 0.5 * (right_asymptote + left_asymptote) - 0.5
    return scale * (1. / (1. + np.exp(-(x - center)))) + shift


def spsa(u, J, c, a, conv_u, conv_J, max_steps=np.inf, alpha=0.602, A=1., gamma=0.101, m=0., verbose=False):
    """
    Uses simulataneous perturbation stochastic approximation [1]_ [2]_ to minimize a function `J` with respect to the
    vector `u`. Notation follows the Wikipedia_ page on the topic at the time of writing.

    At each step we perturb our loss function, :math:`J`, in a random direction :math:`c_n \Delta_n`, and look at the
    first-order approximation to the gradient in that direction. :math:`c_n` is then allowed to shrink each step. It's
    rather like finite difference, but we probe all dimensions simultaneously, so we need only *two* evaluations of
    :math:`J` instead of the number of dimensions evaluations. Pretty nifty.

    Parameter defaults are taken from Spall's 1998 IEEE paper_ [3]_.

    .. note::
        The proof of convergence sketched on Wikipedia_ requires :math:`J` to be thrice differentiable, so this might
        not converge for your problem.

    .. [1] Spall, IEEE Transactions on Automatic Control 37 (1992)
    .. [2] Maryak, Chin, IEEE Transactions on Automatic Control 53 (2008)
    .. [3] Spall, IEEE Transactions on Aerospace and Electronic Systems 24 (1998)
    .. _Wikipedia: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    .. _paper: http://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF

    Args:
        u (np.ndarray): Initial guess for the vector that minimizes the function J.
        J (function): The function to  minimize. Must take `u` as an argument and return a scalar.
        c (float): Initial magnitude of the probing step in `u`-space.
        a (float): Initial magnitude of the gradient scalar. Should be > `c` > 0.
        conv_u (float): Maximum distance of consecutive best-`J` `u`'s to be considered converged.
        conv_J (float): Maximum `J` difference between best-`J`'s to be considered converged.
        max_steps (int): Maximum number of gradient steps to take. (Default is unlimited.)
        alpha (float): Power at which `a` decays with iteration, i.e. :math:`a_n = a_0 / (A + n)^\\gamma.` (Default is
                       0.602)
        A (float): Offset for `a`-decay. (Default is 1.)
        gamma (float): Power at which c decays with iteration, i.e. :math:`c_n = c_0 / n^\\gamma.` (Default is 0.101.)
        m (float): Maximum momentum factor, i.e. fraction of last change to add to new change. Ramps on.
                   (Default is 0.)
        verbose (bool): Whether to print the fit.

    Returns:
        2-element tuple containing

        - (*np.ndarray*) -- Best found solution in `u`-space.
        - (*int*) -- Number of iterations.
    """

    dims = len(u)
    n = 1

    a0 = a
    # TODO: a0 in a similar manner as optimal step size for gradient descent? (Sam's idea)
    c0 = c
    last_du = 0.
    if verbose:
        print ("SPSA step 0 u =" + str(u))
    u_best = u
    J_best = J(u)

    u_best_change = np.inf
    J_best_change = np.inf

    if A > 0:
        a = a0 / A ** alpha
    a1 = a

    while ((u_best_change > conv_u) or (J_best_change > conv_J)) and (n < max_steps):
        delta = rademacher(dims, size=c)
        Jp = J(u + delta)
        Jm = J(u - delta)
        dJ = Jp - Jm
        gu = dJ / (2 * delta)
        force = np.linalg.norm(gu)

        du = -a * gu + m * (1 - (a / a1)) * last_du  # Ramp momentum on as we progress and a/a0 shrinks.
        u += du
        a = a0 / (A + n) ** alpha
        c = c0 / float(n) ** gamma

        losses = [Jp, Jm]
        loss_vecs = [u + delta, u - delta]
        lower_id = np.argmin(losses)
        new_loss = losses[lower_id]
        new_vec = loss_vecs[lower_id]

        if new_loss < J_best:
            u_best_change = np.linalg.norm(new_vec - u_best)
            J_best_change = abs(J_best - new_loss)
            J_best = new_loss
            u_best = new_vec

        if verbose:
            print("Step " + str(n) + ": best J, u, J, F, dub, dJb = " +
                  ", ".join([str(J_best), str(new_vec), str(new_loss), str(force), str(u_best_change),
                             str(J_best_change)]))
        n += 1

    if verbose:
        print ("Best loss" + str(J_best) + " at " + str(u_best))
    return u_best, n


def uniform_points_on_sphere(N, r=1., min_polar=0., max_polar=np.pi, min_azimuthal=0., max_azimuthal=2 * np.pi,
                             inclusive=False):
    """
    Generate points on the surface of a sphere (or portion thereof) in a uniform way. How much of the sphere to use is
    controlled by the maximum polar and azimuthal angles, e.g. `max_polar` = Pi/2 and `max_azithumal` = Pi / 2 will use
    only the +++ octant of the sphere, while `max_polar` = Pi and `max_azithumal` = Pi uses the (+/-)+(+/-) hemisphere.

    Adapted from a 2004 pdf_ from Markus Deserno.

    .. _pdf: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf

    Args:
        N (int): Approximate number of points on the sphere (portion).
        r (float): Radius of the sphere.
        min_polar (float): Minimum polar angle. (Default is 0.)
        max_polar (float): Maximum polar angle (if <2 Pi, only a portion of the sphere is used.) (Default is Pi.)
        min_azimuthal (float): Minimum azimuthal angle. (Default is 0.)
        max_azimuthal (float): Maximum azimuthal angle (if <2 Pi, only a portion of the sphere is used.) (Default is 2
                               Pi.)
        inclusive (bool): Whether to include points at the end of the azimuthal angle. (Default is False.)

    Returns:
        (*np.ndarray*) -- :math:`(n, 3)` vector of equidistributed points on the surface of the unit sphere.
    """
    # polar_end = min_polar + max_polar
    dpolar = max_polar - min_polar
    # azimuthal_end = min_azimuthal + max_azimuthal
    dazimuthal = max_azimuthal - min_azimuthal

    a = 4 * (dazimuthal * dpolar / (2. * np.pi)) * r**2 / float(N)
    d = np.sqrt(a)
    M_theta = np.round(dpolar * r / d)
    d_theta = dpolar * r / float(M_theta)
    d_phi = a / d_theta

    points = []

    for m in np.arange(M_theta):
        theta = (dpolar * (m + 0.5) / float(M_theta)) + min_polar
        M_phi = np.round(dazimuthal * r * np.sin(theta) / d_phi)
        for n in np.arange(M_phi + inclusive):
            phi = (dazimuthal * n / float(M_phi)) + min_azimuthal
            points += [r * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])]

    points = np.array(points)
    return points


def vectors_are_antiparallel(a, b, precision=BASE_PREC):
    """A bool for whether two vectors are anti-parallel."""
    return 1 + _array_dot(a, b) < precision


def vectors_are_parallel(a, b, precision=BASE_PREC):
    """A bool for whether two vectors are parallel."""
    return 1 - _array_dot(a, b) < precision


def vectors_are_perpendicular(a, b, precision=BASE_PREC):
    """A bool for whether two vectors are perpendicular."""
    return abs(_array_dot(a, b)) < precision


def _array_dot(a, b): return np.sum(l2normalize(a) * l2normalize(b))
