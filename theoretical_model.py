import scipy.constants as const
from math import sqrt


def get_estimated_slope(temp, temp_err, visc, visc_err, rad, rad_error):
    """
    Returns the theoretical estimate of the slope coefficient that connects <r^2> to t, with error.
    :param temp: the temperature in Kelvin
    :param temp_err: the error in temperature, in kelvin
    :param visc: the dynamic viscosity in m^2/s
    :param visc_err: the viscosity error.
    :param rad: the radius of the particle in meters.
    :param rad_error: the error in particle radius, in meters
    :return: a tupple in the form (slope, slope_err) such that slope_err is the propagated error. Both return
    values are in m^2/s
    """
    slope = (2 * const.k * temp) / (3 * const.pi * visc * rad)
    slope_err = sqrt((temp_err / temp) ^ 2 + (visc_err / visc) ^ 2 + (rad_error / rad) ^ 2) * slope
    return slope, slope_err