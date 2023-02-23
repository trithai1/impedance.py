import warnings

import numpy as np
from scipy.linalg import inv
from scipy.optimize import curve_fit, basinhopping

from .fitting import set_default_bounds, buildCircuit, wrapCircuit, rmse

ints = '0123456789'


def sb_circuit_fit(frequencies, impedances, circuit, initial_guess, constants={},
                   bounds=None, weight_by_modulus=False, global_opt=False,
                   **kwargs):

    """ Main function for fitting an equivalent circuit to data.
    For detalis refer to impedance.models.circuits.fittting.circuit_fit
    """
    f = np.array(frequencies, dtype=float)
    Z = np.array(impedances, dtype=complex)

    # set upper and lower bounds on a per-element basis
    if bounds is None:
        bounds = set_default_bounds(circuit, constants=constants)

    if not global_opt:
        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = 1e5
        if 'ftol' not in kwargs:
            kwargs['ftol'] = 1e-13

        # weighting scheme for fitting
        if weight_by_modulus:
            abs_Z = np.abs(Z)
            kwargs['sigma'] = np.hstack([abs_Z, abs_Z])

        # built circuit to pass to the optimizer
        cir = buildCircuit(circuit, frequencies, *[f'P{i}' for i in range(len(initial_guess))],
                           constants=constants, eval_string='',
                           index=0)[0]

        if kwargs['fit_bode']:
            kwargs.pop('fit_bode')
            popt, pcov = curve_fit(wrapCircuit(circuit, constants, cir, flag=True), f,
                                   np.hstack([np.abs(Z), np.angle(Z, deg=True)]),
                                   p0=initial_guess, bounds=bounds, **kwargs)

        else:
            kwargs.pop('fit_bode')
            popt, pcov = curve_fit(wrapCircuit(circuit, constants, cir, flag=True), f,
                                   np.hstack([Z.real, Z.imag]),
                                   p0=initial_guess, bounds=bounds, **kwargs)

        # Calculate one standard deviation error estimates for fit parameters,
        # defined as the square root of the diagonal of the covariance matrix.
        # https://stackoverflow.com/a/52275674/5144795
        perror = np.sqrt(np.diag(pcov))

    else:
        if 'seed' not in kwargs:
            kwargs['seed'] = 0

        def opt_function(x):
            """ Short function for basinhopping to optimize over.
            We want to minimize the RMSE between the fit and the data.

            Parameters
            ----------
            x : args
                Parameters for optimization.

            Returns
            -------
            function
                Returns a function (RMSE as a function of parameters).
            """
            return rmse(wrapCircuit(circuit, constants)(f, *x),
                        np.hstack([Z.real, Z.imag]))

        class BasinhoppingBounds(object):
            """ Adapted from the basinhopping documetation
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
            """

            def __init__(self, xmin, xmax):
                self.xmin = np.array(xmin)
                self.xmax = np.array(xmax)

            def __call__(self, **kwargs):
                x = kwargs['x_new']
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin

        basinhopping_bounds = BasinhoppingBounds(xmin=bounds[0],
                                                 xmax=bounds[1])
        results = basinhopping(opt_function, x0=initial_guess,
                               accept_test=basinhopping_bounds, **kwargs)
        popt = results.x

        # Calculate perror
        jac = results.lowest_optimization_result['jac'][np.newaxis]
        try:
            # jacobian -> covariance
            # https://stats.stackexchange.com/q/231868
            pcov = inv(np.dot(jac.T, jac)) * opt_function(popt) ** 2
            # covariance -> perror (one standard deviation
            # error estimates for fit parameters)
            perror = np.sqrt(np.diag(pcov))
        except (ValueError, np.linalg.LinAlgError):
            warnings.warn('Failed to compute perror')
            perror = None

    return popt, perror
