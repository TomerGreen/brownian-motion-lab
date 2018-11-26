import scipy.constants as const
from math import sqrt


PIXEL_LENGTH_IN_METERS = 0.3756*(10**(-6))
PIXEL_LENGTH_ERROR = 0.003*(10**(-6))
SECONDS_PER_FRAME = 0.1
RELATIVE_PIXEL_NUM_ERROR = 0.1  # If this is 0.1, then a size of 12 pixels will have an error of 1.2 pixels.


def get_slope(temp, visc, rad):
    slope = (2 * const.k * temp) / (3 * const.pi * visc * rad)
    return slope


def get_relative_error(temp, temp_err, visc, visc_err, rad, rad_error):
    rel_error = sqrt((temp_err / temp) ^ 2 + (visc_err / visc) ^ 2 + (rad_error / rad) ^ 2)
    return rel_error


def get_estimated_inverse_slope(temp, temp_err, visc, visc_err, rad, rad_error):
    inv_slope = (1/get_slope(temp, visc, rad))
    rel_error = get_relative_error(temp, temp_err, visc, visc_err, rad, rad_error)
    error = rel_error * inv_slope
    return inv_slope, error


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
    slope = get_slope(temp, visc, rad)
    slope_err = get_relative_error(temp, temp_err, visc, visc_err, rad, rad_error) * slope
    return slope, slope_err
