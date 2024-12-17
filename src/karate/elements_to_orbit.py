import jax.numpy as jnp
from jax import jit
from jaxoplanet.core.kepler import kepler


@jit
def true_anomaly_to_eccentric_anomaly(f, ecc):
    """ """
    # make f from -pi to pi
    n = jnp.floor((f + jnp.pi) / (2 * jnp.pi))
    f_norm = (f + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # eccentric anomaly
    u = 2.0 * jnp.arctan(jnp.sqrt((1.0 - ecc) / (1.0 + ecc)) * jnp.tan(f_norm / 2.0))
    u = u + 2.0 * n * jnp.pi
    return jnp.where((f % jnp.pi) == 0, f, u)


@jit
def eccentric_anomaly_to_true_anomaly(u, ecc):
    """ """
    # make u from -pi to pi
    n = jnp.floor((u + jnp.pi) / (2 * jnp.pi))
    u_norm = (u + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # true anomaly
    f = 2.0 * jnp.arctan(jnp.sqrt((1.0 + ecc) / (1.0 - ecc)) * jnp.tan(u_norm / 2.0))
    f = f + 2.0 * n * jnp.pi
    return jnp.where((u % jnp.pi) == 0, u, f)


@jit
def eccentric_anomaly_to_t_from_tperi(u, ecc, period):
    """ """
    return period / (2.0 * jnp.pi) * (u - ecc * jnp.sin(u))


@jit
def tperi_to_t0(t_periastron, period, ecc, omega):
    """ """
    u_t0 = true_anomaly_to_eccentric_anomaly(jnp.pi / 2.0 - omega, ecc)
    t0_from_tperi = eccentric_anomaly_to_t_from_tperi(u_t0, ecc, period)
    return t0_from_tperi + t_periastron


@jit
def t0_to_tperi(t0, period, ecc, omega):
    """ """
    u_t0 = true_anomaly_to_eccentric_anomaly(jnp.pi / 2.0 - omega, ecc)
    t0_from_tperi = eccentric_anomaly_to_t_from_tperi(u_t0, ecc, period)
    return t0 - t0_from_tperi


@jit
def orbital_elements_to_coordinates_circular(t, period, a_over_rs, cosi, t0):
    """
    Compute the x and y coordinates of a planet in a circular orbit.

    This function calculates the position of a planet in a circular
    orbit at a given time `t`. The orbit is described by the orbital
    period, semi-major axis, and inclination.

    Parameters
    ----------
    t : float or array-like
        Time(s) at which to calculate the planet's coordinates.
    period : float
        Orbital period of the planet.
    a_over_rs : float
        Semi-major axis divided by stellar radius.
    cosi : float
        Cosine of the orbital inclination angle.
    t0 : float
        Time of conjunction (when the planet passes closest to the star's center).

    Returns
    -------
    x : float or array-like
        X-coordinate(s) of the planet's position (in units of stellar radii).
    y : float or array-like
        Y-coordinate(s) of the planet's position (in units of stellar radii).
    """
    inputs = [period, a_over_rs, cosi, t0]
    # Process each input: add a new axis if it is an array
    processed_inputs = []
    for inp in inputs:
        arr = jnp.asarray(inp)  # Convert to jax.numpy array
        if arr.ndim > 0:  # Check if it is an array (not a scalar)
            arr = jnp.expand_dims(arr, axis=-1)  # Add a new axis
        processed_inputs.append(arr)
    period, a_over_rs, cosi, t0 = processed_inputs

    x = a_over_rs * jnp.sin(2.0 * jnp.pi * (t - t0) / period)
    y = -a_over_rs * jnp.cos(2.0 * jnp.pi * (t - t0) / period) * cosi
    return x, y


@jit
def orbital_elements_to_coordinates(t, period, a_over_rs, ecc, omega, cosi, t0):
    """
    Compute the x and y coordinates of a planet in a circular orbit.

    This function calculates the position of a planet in a circular
    orbit at a given time `t`. The orbit is described by the orbital
    period, semi-major axis, and inclination.

    Parameters
    ----------
    t : float or 1d array-like
        Time(s) at which to calculate the planet's coordinates.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis divided by stellar radius.
    cosi : float or array-like
        Cosine of the orbital inclination angle.
    t0 : float or array-like
        Time of conjunction (when the planet passes closest to the star's center).

    Returns
    -------
    x : float or array-like (*shape(params), len(t))
        X-coordinate(s) of the planet's position (in units of stellar radii).
    y : float or array-like (*shape(params), len(t))
        Y-coordinate(s) of the planet's position (in units of stellar radii).
    """
    inputs = [period, a_over_rs, ecc, omega, cosi, t0]
    # Process each input: add a new axis if it is an array
    processed_inputs = []
    for inp in inputs:
        arr = jnp.asarray(inp)  # Convert to jax.numpy array
        if arr.ndim > 0:  # Check if it is an array (not a scalar)
            arr = jnp.expand_dims(arr, axis=-1)  # Add a new axis
        processed_inputs.append(arr)
    period, a_over_rs, ecc, omega, cosi, t0 = processed_inputs

    tperi = t0_to_tperi(t0, period, ecc, omega)
    sinf, cosf = get_ta(t, period, ecc, tperi)
    r = a_over_rs * (1.0 - ecc**2) / (1 + ecc * cosf)
    f = jnp.arctan2(sinf, cosf)
    c_X = -r * jnp.cos(omega + f)
    c_Y = -r * jnp.sin(omega + f) * cosi
    return c_X, c_Y


def get_ta(t, period, ecc, tperi):
    """ """
    M = 2 * jnp.pi * (t - tperi) / period
    return kepler(M, ecc)
