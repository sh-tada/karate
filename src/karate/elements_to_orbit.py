import jax.numpy as jnp
from jaxoplanet.core.kepler import kepler


def true_anomaly_to_eccentric_anomaly(f, ecc):
    """
    Convert true anomaly (f) to eccentric anomaly (u) using Kepler's equation.

    Parameters
    ----------
    f : float or array-like
        True anomaly in radians.
    ecc : float or array-like
        Orbital eccentricity.

    Returns
    -------
    u : float or array-like
        Eccentric anomaly in radians.
    """
    # make f from -pi to pi
    n = jnp.floor((f + jnp.pi) / (2 * jnp.pi))
    f_norm = (f + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # eccentric anomaly
    u = 2.0 * jnp.arctan(jnp.sqrt((1.0 - ecc) / (1.0 + ecc)) * jnp.tan(f_norm / 2.0))
    u = u + 2.0 * n * jnp.pi
    return jnp.where((f % jnp.pi) == 0, f, u)


def eccentric_anomaly_to_true_anomaly(u, ecc):
    """
    Convert eccentric anomaly (u) to true anomaly (f) using Kepler's equation.

    Parameters
    ----------
    u : float or array-like
        Eccentric anomaly in radians.
    ecc : float or array-like
        Orbital eccentricity.

    Returns
    -------
    f : float or array-like
        True anomaly in radians.
    """
    # make u from -pi to pi
    n = jnp.floor((u + jnp.pi) / (2 * jnp.pi))
    u_norm = (u + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # true anomaly
    f = 2.0 * jnp.arctan(jnp.sqrt((1.0 + ecc) / (1.0 - ecc)) * jnp.tan(u_norm / 2.0))
    f = f + 2.0 * n * jnp.pi
    return jnp.where((u % jnp.pi) == 0, u, f)


def eccentric_anomaly_to_t_from_tperi(u, ecc, period):
    """
    Convert eccentric anomaly (u) to time (t) from periastron (t_periastron)
    based on orbital parameters.

    Parameters
    ----------
    u : float or array-like
        Eccentric anomaly in radians.
    ecc : float or array-like
        Orbital eccentricity.
    period : float or array-like
        Orbital period of the planet.

    Returns
    -------
    t : float or array-like
        Time (t) at which the planet is at the given eccentric anomaly.
    """
    return period / (2.0 * jnp.pi) * (u - ecc * jnp.sin(u))


def tperi_to_t0(t_periastron, period, ecc, omega):
    """
    Calculate the time of inferior conjunction (t0) from the time of periastron
    (t_periastron).

    Parameters
    ----------
    t_periastron : float
        Time of periastron passage.
    period : float
        Orbital period of the planet.
    ecc : float
        Orbital eccentricity.
    omega : float
        Argument of periastron in radians.

    Returns
    -------
    t0 : float
        Time of inferior conjunction.
    """
    u_t0 = true_anomaly_to_eccentric_anomaly(jnp.pi / 2.0 - omega, ecc)
    t0_from_tperi = eccentric_anomaly_to_t_from_tperi(u_t0, ecc, period)
    return t0_from_tperi + t_periastron


def t0_to_tperi(t0, period, ecc, omega):
    """
    Calculate the time of periastron (t_periastron) from the time of
    inferior conjunction (t0).

    Parameters
    ----------
    t0 : float
        Time of inferior conjunction.
    period : float
        Orbital period of the planet.
    ecc : float
        Orbital eccentricity.
    omega : float
        Argument of periastron in radians.

    Returns
    -------
    t_periastron : float
        Time of periastron passage.
    """
    u_t0 = true_anomaly_to_eccentric_anomaly(jnp.pi / 2.0 - omega, ecc)
    t0_from_tperi = eccentric_anomaly_to_t_from_tperi(u_t0, ecc, period)
    return t0 - t0_from_tperi


def orbital_elements_to_coordinates_circular(t, period, a_over_rs, cosi, t0):
    """
    Compute the x and y coordinates of a planet in a circular orbit at a given
    time.

    This function calculates the position of a planet in a circular orbit
    at time `t` based on orbital parameters.

    Parameters
    ----------
    t : float or array-like
        Time(s) at which to calculate the planet's coordinates.
    period : float
        Orbital period of the planet.
    a_over_rs : float
        Semi-major axis normalized by the stellar radius.
    cosi : float
        Cosine of the orbital inclination angle.
    t0 : float
        Time of inferior conjunction.

    Returns
    -------
    x : float or array-like
        X-coordinate(s) of the planet's position (in units of stellar radii).
    y : float or array-like
        Y-coordinate(s) of the planet's position (in units of stellar radii).
    """
    inputs = [period, a_over_rs, cosi, t0]
    arrays = [jnp.asarray(inp) for inp in inputs]
    # Process each input: add a new axis if it is an array
    processed_inputs = []
    max_shape = jnp.asarray(1)
    for arr in arrays:
        max_shape = max_shape * jnp.ones_like(arr)
    for arr in arrays:
        arr = arr * max_shape  # Convert to jax.numpy array
        if arr.ndim > 0 and arr.shape[-1] > 1:  # Check if it is an array (not a scalar)
            arr = jnp.expand_dims(arr, axis=-1)  # Add a new axis
        processed_inputs.append(arr)
    period, a_over_rs, cosi, t0 = processed_inputs

    x = a_over_rs * jnp.sin(2.0 * jnp.pi * (t - t0) / period)
    y = -a_over_rs * jnp.cos(2.0 * jnp.pi * (t - t0) / period) * cosi
    return x, y


def orbital_elements_to_coordinates(t, period, a_over_rs, ecc, omega, cosi, t0):
    """
    Compute the x and y coordinates of a planet in an eccentric orbit at a
    given time.

    This function calculates the position of a planet in an eccentric orbit
    at time `t` based on orbital parameters including eccentricity and argument
    of periastron.

    Parameters
    ----------
    t : float or 1d array-like
        Time(s) at which to calculate the planet's coordinates.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by the stellar radius.
    ecc : float or array-like
        Orbital eccentricity.
    omega : float or array-like
        Argument of periastron in radians.
    cosi : float or array-like
        Cosine of the orbital inclination angle.
    t0 : float or array-like
        Time of inferior conjunction.

    Returns
    -------
    x : float or array-like (*shape(params), len(t))
        X-coordinate(s) of the planet's position (in units of stellar radii).
    y : float or array-like (*shape(params), len(t))
        Y-coordinate(s) of the planet's position (in units of stellar radii).
    """
    inputs = [period, a_over_rs, ecc, omega, cosi, t0]
    arrays = [jnp.asarray(inp) for inp in inputs]
    # Process each input: add a new axis if it is an array
    processed_inputs = []
    max_shape = jnp.asarray(1)
    for arr in arrays:
        max_shape = max_shape * jnp.ones_like(arr)
    for arr in arrays:
        arr = arr * max_shape  # Convert to jax.numpy array
        if arr.ndim > 0 and arr.shape[-1] > 1:  # Check if it is an array (not a scalar)
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
    """
    Calculate the mean anomaly (M) and use it to solve Kepler's equation for
    the true anomaly (f).

    Parameters
    ----------
    t : float or array-like
        Time(s) at which to calculate the true anomaly.
    period : float
        Orbital period of the planet.
    ecc : float
        Orbital eccentricity.
    tperi : float
        Time of periastron passage.

    Returns
    -------
    sinf : float or array-like
        Sine of the true anomaly.
    cosf : float or array-like
        Cosine of the true anomaly.
    """
    M = 2 * jnp.pi * (t - tperi) / period
    return kepler(M, ecc)
